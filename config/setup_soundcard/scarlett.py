"""
Control the Scarlett 4th Gen (18i16 / 18i20) preamps (gain, phantom, air, ...) and
line output volumes from the command line, through the ALSA controls created by
fcp-server.

The controls ("Line In <n> Gain Capture Volume", ...) only exist when:
    - the kernel is 6.14 or later (FCP driver), and
    - fcp-server is running for the card (started by udev/systemd).
Run `install.sh` once to set that up, then `status` to check every layer.

Usage (from the project root):
    python config/setup_soundcard/scarlett.py cards             # ALSA cards, Focusrite marked
    python config/setup_soundcard/scarlett.py status            # what is installed / running
    python config/setup_soundcard/scarlett.py show              # current input/output settings
    python config/setup_soundcard/scarlett.py apply             # apply gains.yaml
    python config/setup_soundcard/scarlett.py apply -f other.yaml
    python config/setup_soundcard/scarlett.py set 3 --gain 45 --phantom on
    python config/setup_soundcard/scarlett.py set all --gain 30
    python config/setup_soundcard/scarlett.py set-output 1 --volume -20
    python config/setup_soundcard/scarlett.py set-master --volume -10 --mute off

The Focusrite card is found automatically (only one is expected to be connected);
use --card <number|id> to force another one.
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_PRESET = HERE / "gains.yaml"
FOCUSRITE = re.compile(r"Scarlett|Focusrite", re.I)

# preset key -> (ALSA control name, kind), in the order they are written
# (input autogain/safe go first so they cannot override the gain)
CONTROLS = {
    "autogain": ("Line In {n} Autogain Capture Switch", "bool"),
    "safe": ("Line In {n} Safe Capture Switch", "bool"),
    "level": ("Line In {n} Level Capture Enum", "enum"),
    "air": ("Line In {n} Air Capture Enum", "enum"),
    "phantom": ("Line In {n} Phantom Power Capture Switch", "bool"),
    "gain": ("Line In {n} Gain Capture Volume", "int"),
}
OUTPUT_CONTROLS = {
    "mute": ("Line {n} Mute Playback Switch", "bool"),
    "volume": ("Analogue {n} Playback Volume", "int"),
}
MASTER_CONTROLS = {
    "mute": ("Mute Playback Switch", "bool"),
    "dim": ("Dim Playback Switch", "bool"),
    "volume": ("Master Playback Volume", "int"),
}
# allowed ranges of the int settings
RANGES = {"gain": (0, 70), "volume": (-127, 0)}


# ---------------------------------------------------------------------------
# Card discovery
# ---------------------------------------------------------------------------
def list_cards():
    """ALSA cards as (number, id, description), from /proc/asound/cards."""
    text = Path("/proc/asound/cards").read_text()
    return [
        (num, cid, desc.strip())
        for num, cid, desc in re.findall(r"^\s*(\d+)\s+\[(\S+)\s*\]:\s*(.*)$", text, re.M)
    ]


def print_cards(cards):
    if not cards:
        print("  (no ALSA cards)")
    for num, cid, desc in cards:
        mark = "*" if FOCUSRITE.search(desc) else " "
        print(f" {mark} {num:>2}  {cid:<12} {desc}")


def find_card(requested=None):
    """ALSA card number to use: `requested` (number or id) or the only Focusrite card."""
    cards = list_cards()
    if requested is not None and str(requested).lower() != "auto":
        for num, cid, _ in cards:
            if str(requested) in (num, cid):
                return num
        print(f"card '{requested}' not found. ALSA cards (* = Focusrite):")
        print_cards(cards)
        raise SystemExit(1)

    focusrite = [c for c in cards if FOCUSRITE.search(c[2])]
    if len(focusrite) == 1:
        num, cid, desc = focusrite[0]
        print(f"using card {num} [{cid}] {desc}")
        return num
    if not focusrite:
        print("no Focusrite card found (is it plugged in and powered?). ALSA cards:")
    else:
        print("more than one Focusrite card: choose one with --card <number>. ALSA cards:")
    print_cards(cards)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# amixer helpers
# ---------------------------------------------------------------------------
def amixer(card, *args, check=True):
    res = subprocess.run(
        ["amixer", "-c", str(card), *args], capture_output=True, text=True
    )
    if check and res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip())
    return res


def control_names(card):
    """Names of all the ALSA controls of the card."""
    out = amixer(card, "controls").stdout
    return set(re.findall(r"name='([^']*)'", out))


def cget(card, name):
    """Current value of a control, decoded (int, bool or enum item name)."""
    out = amixer(card, "cget", f"name={name}").stdout
    kind = re.search(r"type=(\w+)", out).group(1)
    value = re.search(r": values=([^\n]*)", out).group(1).split(",")[0]
    if kind == "BOOLEAN":
        return value == "on"
    if kind == "ENUMERATED":
        items = dict(re.findall(r"Item #(\d+) '([^']*)'", out))
        return items.get(value, value)
    return int(value)


def cset(card, name, kind, value):
    if kind == "bool":
        value = "on" if value else "off"
    elif kind == "int":
        value = str(int(value))
    amixer(card, "cset", "--", f"name={name}", str(value))  # "--": negative dB values


def parse_bool(text):
    t = str(text).lower()
    if t in ("1", "on", "true", "yes"):
        return True
    if t in ("0", "off", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got '{text}'")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def count(controls, key, names):
    """Number of channels exposing controls[key] (inputs: 4 on 18i16 / 8 on 18i20)."""
    n = 0
    while controls[key][0].format(n=n + 1) in names:
        n += 1
    return n


def n_inputs(names):
    return count(CONTROLS, "gain", names)


def n_outputs(names):
    return count(OUTPUT_CONTROLS, "volume", names)


def require_controls(card):
    names = control_names(card)
    if not n_inputs(names):
        raise SystemExit(
            f"card '{card}' has no 'Line In <n> Gain' controls: fcp-server is not running "
            "or the kernel has no FCP driver.\nRun: python config/setup_soundcard/scarlett.py status"
        )
    return names


def print_table(card, names, label, controls, n_channels):
    keys = list(controls)
    print(f"{label:>6}  " + "  ".join(f"{k:>10}" for k in keys))
    for n in range(1, n_channels + 1):
        cells = []
        for key in keys:
            name = controls[key][0].format(n=n)
            cells.append(f"{str(cget(card, name)) if name in names else '-':>10}")
        print(f"{n:>6}  " + "  ".join(cells))


def show(card):
    names = require_controls(card)
    print_table(card, names, "input", CONTROLS, n_inputs(names))
    print()
    print_table(card, names, "output", OUTPUT_CONTROLS, n_outputs(names))
    print()
    print("master  " + "  ".join(
        f"{key} = {cget(card, name)}" for key, (name, _) in MASTER_CONTROLS.items() if name in names
    ))


def apply_settings(card, label, settings, names, controls, n=None):
    """Write `settings` ({key: value}) of one channel (n) or of the master section."""
    for key in sorted(settings, key=lambda k: list(controls).index(k) if k in controls else -1):
        if key not in controls:
            raise SystemExit(f"{label}: unknown setting '{key}' (valid: {', '.join(controls)})")
        name, kind = controls[key]
        name = name.format(n=n)
        if name not in names:
            print(f"  {label}: '{key}' not available on this device, skipped")
            continue
        value = settings[key]
        if key in RANGES:
            lo, hi = RANGES[key]
            if not lo <= int(value) <= hi:
                raise SystemExit(f"{label}: {key} {value} out of range {lo}..{hi}")
        cset(card, name, kind, value)
        print(f"  {label}: {key} = {cget(card, name)}")


def apply_channels(card, names, kind, channels, controls, n_channels):
    """Apply {n: settings} to inputs or outputs; n is a channel number or 'all'."""
    for which, settings in channels.items():
        if str(which) == "all":
            todo = range(1, n_channels + 1)
        elif 1 <= int(which) <= n_channels:
            todo = [int(which)]
        else:
            raise SystemExit(f"{kind} {which} out of range 1-{n_channels}")
        for n in todo:
            apply_settings(card, f"{kind} {n}", settings or {}, names, controls, n)


def apply_preset(path, card_override=None):
    with open(path, encoding="utf-8") as fh:
        preset = yaml.safe_load(fh)
    card = find_card(card_override or preset.get("card"))
    names = require_controls(card)
    print(f"applying {path} to card {card}")
    apply_channels(card, names, "input", preset.get("inputs") or {}, CONTROLS, n_inputs(names))
    apply_channels(
        card, names, "output", preset.get("outputs") or {}, OUTPUT_CONTROLS, n_outputs(names)
    )
    if preset.get("master"):
        apply_settings(card, "master", preset["master"], names, MASTER_CONTROLS)


def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return (res.stdout + res.stderr).strip()


def status(card):
    """Check every layer needed for the gain controls, from the kernel up."""

    def line(ok, text, hint=None):
        print(f"[{'OK' if ok else '!!'}] {text}")
        if not ok and hint:
            print(f"     -> {hint}")
        return ok

    release = platform.release()
    major, minor = (int(x) for x in re.match(r"(\d+)\.(\d+)", release).groups())
    line(
        (major, minor) >= (6, 14),
        f"kernel {release} (FCP driver needs >= 6.14)",
        "./config/setup_soundcard/install.sh kernel   (then reboot)",
    )

    try:
        num = find_card(card)
    except SystemExit:
        line(False, "Focusrite ALSA card present", "plug in the Scarlett / check aplay -l")
        return
    line(True, f"ALSA card {num} present")
    card = num

    for tool in ("fcp-server", "fcp-tool", "alsa-scarlett-gui"):
        line(bool(shutil.which(tool)), f"{tool} installed", "./config/setup_soundcard/install.sh")

    active = run(f"systemctl is-active fcp-server@{num}.service")
    line(
        active == "active",
        f"fcp-server@{num}.service: {active}",
        f"journalctl -u fcp-server@{num} -b   (or: LOG_LEVEL=debug fcp-server {num})",
    )

    groups = run("id -nG").split()
    line("audio" in groups, "user in group 'audio'", "sudo usermod -aG audio $USER, then log out/in")

    names = control_names(card)
    line(
        CONTROLS["gain"][0].format(n=1) in names,
        f"gain controls exposed ({len(names)} ALSA controls on the card)",
        "they appear once fcp-server is running on a >= 6.14 kernel",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--card", default=None, help="ALSA card number or id (default: the only Focusrite card)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cards", help="list the ALSA cards (Focusrite marked with *)")
    sub.add_parser("status", help="check kernel, fcp-server and controls")
    sub.add_parser("show", help="print the current input, output and master settings")

    pa = sub.add_parser("apply", help="apply a preset file")
    pa.add_argument("-f", "--file", type=Path, default=DEFAULT_PRESET)

    ps = sub.add_parser("set", help="change single input settings")
    ps.add_argument("input", help="input number (1-4 on 18i16, 1-8 on 18i20), or 'all'")
    ps.add_argument("--gain", type=int)
    ps.add_argument("--phantom", type=parse_bool)
    ps.add_argument("--air", choices=["Off", "Presence", "Presence + Drive"])
    ps.add_argument("--level", choices=["Line", "Inst"])
    ps.add_argument("--autogain", type=parse_bool)
    ps.add_argument("--safe", type=parse_bool)

    po = sub.add_parser("set-output", help="change line output volume / mute")
    po.add_argument("output", help="line output number (1-8), or 'all'")
    po.add_argument("--volume", type=int, help="dB, -127..0")
    po.add_argument("--mute", type=parse_bool)

    pm = sub.add_parser("set-master", help="change master volume / mute / dim")
    pm.add_argument("--volume", type=int, help="dB, -127..0")
    pm.add_argument("--mute", type=parse_bool)
    pm.add_argument("--dim", type=parse_bool)

    args = p.parse_args()
    if args.cmd == "cards":
        print("ALSA cards (* = Focusrite):")
        print_cards(list_cards())
        return
    if not shutil.which("amixer"):
        raise SystemExit("amixer not found: sudo apt install alsa-utils")

    try:
        if args.cmd == "status":
            status(args.card)
        elif args.cmd == "show":
            show(find_card(args.card))
        elif args.cmd == "apply":
            apply_preset(args.file, args.card)
        elif args.cmd in ("set", "set-output", "set-master"):
            controls = {"set": CONTROLS, "set-output": OUTPUT_CONTROLS}.get(args.cmd, MASTER_CONTROLS)
            settings = {k: getattr(args, k) for k in controls if getattr(args, k) is not None}
            if not settings:
                raise SystemExit(f"nothing to set (use {', '.join('--' + k for k in controls)})")
            card = find_card(args.card)
            names = require_controls(card)
            if args.cmd == "set":
                apply_channels(card, names, "input", {args.input: settings}, CONTROLS, n_inputs(names))
            elif args.cmd == "set-output":
                apply_channels(
                    card, names, "output", {args.output: settings}, OUTPUT_CONTROLS, n_outputs(names)
                )
            else:
                apply_settings(card, "master", settings, names, MASTER_CONTROLS)
    except RuntimeError as e:
        raise SystemExit(f"amixer error: {e}")


if __name__ == "__main__":
    main()
