"""
Sweep measurement: play one sine sweep and record a reference mic plus a mic
under test, then compare their power spectral densities.

Pipeline:
    1. generate a sine sweep (linear like PlayTone.make_sweep, or log)
    2. play it on one output channel while recording the reference channel
       (ch1 by default) and the mic channel under test
    3. locate the sweep in the reference recording (cross-correlation) and
       cut the same window from both channels; the pre-roll silence is kept
       as a noise-floor estimate
    4. compute the Welch PSD of both channels and plot them together with
       their difference (mic - ref, dB) over the swept band

Channels are 1-based (ch1 = first input of the soundcard).

Outputs go to tests/measurements/results/<message>/, where <message> is the
test name given with -m (default "Test <n>", n = next free number):
    recording.wav   2 channels: [reference, mic], float32
    sweep.wav       the played sweep
    psd.csv         freq, PSD ref/mic (dB), difference (dB), noise floors
    psd.png         the plot
    meta.json       measurement parameters (used again by --analyze)

The ALSA backend needs pyalsaaudio, which only exists on Linux: run it inside
the BAT conda env on the Pi, like main.py (conda activate BAT). On other
systems the default backend is sounddevice.

Usage (from the project root):
    # Raspberry Pi + Scarlett through ALSA (same device as PlayTone)
    python tests/measurements/sweep_psd.py --mic-ch 3 --f-start 5000 --f-end 90000 --duration 500 -m "mic3 front"

    # Windows: sounddevice, picks the Focusrite WDM-KS input/output pair by itself
    python tests/measurements/sweep_psd.py --mic-ch 3 -m "mic3 front"

    # any other soundcard: input and output device by index or name (see --list-devices)
    python tests/measurements/sweep_psd.py --in-device 78 --out-device 79 --mic-ch 3

    # recompute the plot from an existing recording
    python tests/measurements/sweep_psd.py --analyze tests/measurements/results/<folder>
"""

import argparse
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

MEASUREMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MEASUREMENTS_DIR / "results"

INT32_FS = 2**31 - 1


# ---------------------------------------------------------------------------
# Sweep generation
# ---------------------------------------------------------------------------
def make_sweep(
    f_start, f_end, duration_ms, fs, amplitude=0.5, method="linear", fade=0.05
):
    """Generate a sine sweep as float32 in [-amplitude, amplitude].

    Args:
        f_start, f_end: boundary frequencies [Hz]
        duration_ms: sweep length [ms]
        fs: sample rate [Hz]
        amplitude: peak amplitude, full scale = 1.0
        method: "linear" (same phase law as PlayTone.make_sweep) or "log"
        fade: fraction of the sweep used by EACH of the fade-in / fade-out
              linear ramps. 0.5 reproduces the triangular PlayTone envelope;
              small values keep the spectrum flatter over the band.
    """
    n = int(round(fs * duration_ms / 1000))
    t = np.arange(n) / fs
    sweep = signal.chirp(
        t,
        f0=f_start,
        t1=t[-1],
        f1=f_end,
        method="linear" if method == "linear" else "logarithmic",
        phi=-90,
    )

    ramp = int(n * min(max(fade, 0.0), 0.5))
    window = np.ones(n)
    if ramp > 0:
        window[:ramp] = np.linspace(0, 1, ramp)
        window[-ramp:] = np.linspace(1, 0, ramp)

    return (amplitude * sweep * window).astype(np.float32)


# ---------------------------------------------------------------------------
# Play + record backends
# ---------------------------------------------------------------------------
def resolve_sd_device(arg, kind):
    """Device index from an index or a (partial, case-insensitive) name. kind: "input" | "output"."""
    import sounddevice as sd

    if arg is None or str(arg).isdigit():
        return None if arg is None else int(arg)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for idx, dev in enumerate(sd.query_devices()):
        if str(arg).lower() in dev["name"].lower() and dev[key] > 0:
            return idx
    raise SystemExit(f"no {kind} device matching '{arg}' (see --list-devices)")


def find_focusrite_wdmks():
    """(input, output) indices of the Focusrite WDM-KS pair, or None.

    On Windows the Focusrite driver shows its WDM-KS endpoints as
    "Analogue 1 + 2 (wcXXXX)" and "Speakers (wrXXXX)" with the same XXXX.
    WDM-KS talks to the driver directly at its native rate; MME / DirectSound /
    WASAPI-shared go through the Windows mixer, which resamples and cuts
    everything above the Windows device rate (e.g. 24 kHz at 48 kHz).
    """
    import sounddevice as sd

    apis = [i for i, a in enumerate(sd.query_hostapis()) if "WDM-KS" in a["name"]]
    if not apis:
        return None
    devices = [
        (i, d) for i, d in enumerate(sd.query_devices()) if d["hostapi"] == apis[0]
    ]
    for i_in, d_in in devices:
        m = re.search(r"\(wc(\w+)\)", d_in["name"])
        if d_in["max_input_channels"] > 0 and m:
            for i_out, d_out in devices:
                if (
                    f"(wr{m.group(1)})" in d_out["name"]
                    and d_out["max_output_channels"] > 0
                ):
                    print(
                        f"found Focusrite WDM-KS pair: input [{i_in}] {d_in['name']}, output [{i_out}] {d_out['name']}"
                    )
                    return i_in, i_out
    return None


def playrec_sounddevice(
    sweep, fs, out_ch, in_chs, in_device, out_device, pre_s, post_s
):
    """Play `sweep` on out_ch and record in_chs (1-based) through PortAudio."""
    import sounddevice as sd

    pre = np.zeros(int(pre_s * fs), dtype=np.float32)
    post = np.zeros(int(post_s * fs), dtype=np.float32)
    out = np.concatenate([pre, sweep, post])[:, None]

    in_device = resolve_sd_device(in_device, "input")
    out_device = resolve_sd_device(out_device, "output")
    if in_device is None and out_device is None and sys.platform == "win32":
        pair = find_focusrite_wdmks()
        if pair:
            in_device, out_device = pair
    if in_device is None:
        in_device = sd.default.device[0]
    if out_device is None:
        out_device = sd.default.device[1]

    apis = sd.query_hostapis()
    for kind, idx, ch_key in (
        ("input", in_device, "max_input_channels"),
        ("output", out_device, "max_output_channels"),
    ):
        dev = sd.query_devices(idx)
        api = apis[dev["hostapi"]]["name"]
        print(f"sounddevice {kind:6}: [{idx}] {dev['name']}, {api} ({dev[ch_key]} ch)")
        if api in ("MME", "Windows DirectSound"):
            print(
                f"WARNING: {api} resamples through the Windows mixer, ultrasonic content may be cut"
            )
    if out_ch > sd.query_devices(out_device)["max_output_channels"]:
        raise SystemExit(
            f"--out-ch {out_ch} not available on output device [{out_device}]"
        )
    if max(in_chs) > sd.query_devices(in_device)["max_input_channels"]:
        raise SystemExit(
            f"input ch{max(in_chs)} not available on input device [{in_device}]"
        )

    rec = sd.playrec(
        out,
        samplerate=fs,
        device=(in_device, out_device),
        input_mapping=in_chs,
        output_mapping=[out_ch],
        dtype="float32",
        blocking=True,
    )
    return rec


def playrec_alsa(
    sweep,
    fs,
    out_ch,
    in_chs,
    device,
    pre_s,
    post_s,
    blocksize,
    in_channels,
    out_channels,
):
    """Play `sweep` on out_ch and record in_chs (1-based) through ALSA.

    Mirrors PlayTone / AlsaAudioSource: S32_LE interleaved frames on the same
    hw device, playback written block by block while a thread captures.
    """
    try:
        import alsaaudio
    except ImportError as e:
        raise SystemExit(
            f"cannot import alsaaudio ({e}) with interpreter {sys.executable}.\n"
            "pyalsaaudio is Linux only: run this script in the BAT env on the Pi "
            "(conda activate BAT), or use --backend sounddevice."
        )

    capture = alsaaudio.PCM(
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        rate=fs,
        channels=in_channels,
        format=alsaaudio.PCM_FORMAT_S32_LE,
        periodsize=blocksize,
        periods=10,
        device=device,
    )
    playback = alsaaudio.PCM(
        type=alsaaudio.PCM_PLAYBACK,
        mode=alsaaudio.PCM_NORMAL,
        rate=fs,
        channels=out_channels,
        format=alsaaudio.PCM_FORMAT_S32_LE,
        periodsize=blocksize,
        periods=10,
        device=device,
    )

    # pyalsaaudio silently ignores unsupported settings and keeps what the card
    # accepts: read the real values back and build the frames from those
    cinfo, pinfo = capture.info(), playback.info()
    for name, info in (("capture", cinfo), ("playback", pinfo)):
        print(
            f"ALSA {name} {device}: {info.get('channels')} ch, {info.get('rate')} Hz, "
            f"{info.get('format_name')}, period {info.get('period_size')}, buffer {info.get('buffer_size')}"
        )
    in_n, out_n = cinfo["channels"], pinfo["channels"]
    if (in_n, out_n) != (in_channels, out_channels):
        print(
            f"WARNING: requested {in_channels} in / {out_channels} out channels, card opened {in_n} / {out_n}: using those"
        )
    if pinfo.get("rate") != fs or cinfo.get("rate") != fs:
        print(f"WARNING: card is not running at {fs} Hz")
    if out_ch > out_n or max(in_chs) > in_n:
        raise SystemExit(
            f"channel out of range: card has {in_n} inputs and {out_n} outputs"
        )
    out_period = pinfo.get("period_size") or blocksize

    blocks = []
    stop = threading.Event()

    def record():
        while not stop.is_set():
            size, data = capture.read()
            if size > 0:
                blocks.append(
                    np.frombuffer(data, dtype=np.int32).reshape(-1, in_n).copy()
                )

    rec_thread = threading.Thread(target=record, daemon=True)
    rec_thread.start()

    # the whole output (silence + sweep + silence) is written to the card, so the
    # playback stream runs continuously for the full recording
    out = np.concatenate([np.zeros(int(pre_s * fs)), sweep, np.zeros(int(post_s * fs))])
    n_blocks = int(np.ceil(len(out) / out_period))
    tone = np.zeros(n_blocks * out_period, dtype=np.int32)
    # clip before the cast: values beyond full scale would wrap around in int32
    tone[: len(out)] = (np.clip(out, -1.0, 1.0) * INT32_FS).astype(np.int32)

    underruns = 0
    for b in range(n_blocks):
        frame = np.zeros((out_period, out_n), dtype=np.int32)
        frame[:, out_ch - 1] = tone[b * out_period : (b + 1) * out_period]
        written = playback.write(frame.tobytes())
        if written != out_period:
            underruns += 1
    if underruns:
        print(f"WARNING: {underruns} playback writes incomplete (underrun)")

    # wait until everything written has actually been played
    if hasattr(playback, "drain"):
        playback.drain()
    else:
        time.sleep(pinfo.get("buffer_size", 0) / fs)
    time.sleep(0.05)
    stop.set()
    rec_thread.join(timeout=1)
    playback.close()
    capture.close()

    rec = np.concatenate(blocks, axis=0)
    return rec[:, [c - 1 for c in in_chs]].astype(np.float32) / INT32_FS


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def locate_sweep(ref, sweep):
    """Return the sample index where the sweep starts in the reference recording."""
    corr = signal.correlate(ref, sweep, mode="valid", method="fft")
    return int(np.argmax(np.abs(corr)))


def psd_db(x, fs, nperseg):
    f, p = signal.welch(
        x, fs=fs, window="hann", nperseg=min(nperseg, len(x)), scaling="density"
    )
    return f, 10 * np.log10(p + 1e-30)


def analyze(rec, sweep, meta, out_dir):
    fs = meta["fs"]
    nperseg = meta["nperseg"]
    ref, mic = rec[:, 0], rec[:, 1]

    start = locate_sweep(ref, sweep)
    stop = start + len(sweep)
    print(f"sweep found at {start / fs * 1000:.1f} ms in the recording")

    f, ref_db = psd_db(ref[start:stop], fs, nperseg)
    _, mic_db = psd_db(mic[start:stop], fs, nperseg)

    # noise floor from the silence before the sweep (if long enough)
    # (some drivers, e.g. WDM-KS, deliver exact zeros while the input starts up: skip them)
    nonzero = np.flatnonzero((ref[:start] != 0) | (mic[:start] != 0))
    noise_start = nonzero[0] if len(nonzero) else start
    noise_stop = start - int(0.01 * fs)
    has_noise = noise_stop - noise_start >= nperseg
    if has_noise:
        _, ref_noise_db = psd_db(ref[noise_start:noise_stop], fs, nperseg)
        _, mic_noise_db = psd_db(mic[noise_start:noise_stop], fs, nperseg)
    else:
        ref_noise_db = mic_noise_db = np.full_like(f, np.nan)

    diff_db = mic_db - ref_db

    f_lo, f_hi = sorted((meta["f_start"], meta["f_end"]))
    band = (f >= f_lo) & (f <= f_hi)
    print(
        f"in band {f_lo/1000:.1f}-{f_hi/1000:.1f} kHz: "
        f"mic - ref mean {np.mean(diff_db[band]):+.2f} dB, "
        f"min {np.min(diff_db[band]):+.2f} dB, max {np.max(diff_db[band]):+.2f} dB"
    )

    np.savetxt(
        out_dir / "psd.csv",
        np.column_stack([f, ref_db, mic_db, diff_db, ref_noise_db, mic_noise_db]),
        delimiter=",",
        header="freq_hz,ref_db,mic_db,mic_minus_ref_db,ref_noise_db,mic_noise_db",
        comments="",
        fmt="%.6g",
    )

    ref_label = f"reference (ch{meta['ref_ch']})"
    mic_label = f"mic (ch{meta['mic_ch']})"
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), height_ratios=[2, 1]
    )

    ax1.plot(f / 1000, ref_db, color="tab:blue", label=ref_label)
    ax1.plot(f / 1000, mic_db, color="tab:orange", label=mic_label)
    if has_noise:
        ax1.plot(
            f / 1000,
            ref_noise_db,
            color="tab:blue",
            ls=":",
            lw=1,
            label=f"{ref_label} noise",
        )
        ax1.plot(
            f / 1000,
            mic_noise_db,
            color="tab:orange",
            ls=":",
            lw=1,
            label=f"{mic_label} noise",
        )
    ax1.axvspan(f_lo / 1000, f_hi / 1000, color="0.9", zorder=0, label="swept band")
    ax1.set_ylabel("PSD [dB re FS²/Hz]")
    ax1.set_title(
        (f"{meta['message']}\n" if meta.get("message") else "")
        + f"Sweep {meta['f_start']/1000:g}-{meta['f_end']/1000:g} kHz, "
        f"{meta['duration_ms']:g} ms ({meta['method']}), fs = {fs/1000:g} kHz"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize="small")

    ax2.plot(f[band] / 1000, diff_db[band], color="tab:green")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("Frequency [kHz]")
    ax2.set_ylabel("mic - ref [dB]")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, fs / 2000)

    fig.tight_layout()
    fig.savefig(out_dir / "psd.png", dpi=150)
    print(f"results saved in {out_dir}")
    if meta.get("show", True):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_test_dir(message=None):
    """Create the results folder named after the test message.

    No message -> "Test <n>" with n one above the highest existing "Test <n>".
    An existing name gets a _1, _2, ... suffix so earlier tests are never overwritten.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not message:
        numbers = [
            int(m.group(1))
            for d in RESULTS_DIR.iterdir()
            if (m := re.fullmatch(r"Test (\d+)(?:_\d+)?", d.name))
        ]
        message = f"Test {max(numbers, default=0) + 1}"

    # keep the name usable as a folder on any OS
    name = re.sub(r'[<>:"/\\|?*]', "_", message).strip(" .") or "Test"
    out_dir = RESULTS_DIR / name
    suffix = 1
    while out_dir.exists():
        out_dir = RESULTS_DIR / f"{name}_{suffix}"
        suffix += 1
    out_dir.mkdir()
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-m",
        "--message",
        default=None,
        help='test name, used as results folder name (default: "Test <next number>")',
    )
    p.add_argument(
        "--mic-ch",
        type=int,
        default=2,
        help="input channel of the mic under test (default: 2)",
    )
    p.add_argument(
        "--ref-ch",
        type=int,
        default=1,
        help="input channel of the reference mic (default: 1)",
    )
    p.add_argument(
        "--out-ch",
        type=int,
        default=1,
        help="output channel playing the sweep (default: 1)",
    )
    p.add_argument(
        "--f-start",
        type=float,
        default=5000,
        help="sweep start frequency [Hz] (default: 5000)",
    )
    p.add_argument(
        "--f-end",
        type=float,
        default=90000,
        help="sweep end frequency [Hz] (default: 90000)",
    )
    p.add_argument(
        "--duration", type=float, default=500, help="sweep length [ms] (default: 500)"
    )
    p.add_argument(
        "--method",
        choices=["linear", "log"],
        default="linear",
        help="sweep law (default: linear)",
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="peak amplitude, 1.0 = full scale (default: 0.5)",
    )
    p.add_argument(
        "--fade",
        type=float,
        default=0.05,
        help="fade-in/out fraction of the sweep, 0.5 = PlayTone triangle (default: 0.05)",
    )
    p.add_argument(
        "--fs", type=int, default=192000, help="sample rate [Hz] (default: 192000)"
    )
    p.add_argument(
        "--pre",
        type=float,
        default=0.5,
        help="silence recorded before the sweep [s] (default: 0.5)",
    )
    p.add_argument(
        "--post",
        type=float,
        default=0.3,
        help="silence recorded after the sweep [s] (default: 0.3)",
    )
    p.add_argument(
        "--nperseg", type=int, default=2048, help="Welch segment length (default: 2048)"
    )
    p.add_argument(
        "--backend",
        choices=["alsa", "sounddevice"],
        default="alsa" if sys.platform.startswith("linux") else "sounddevice",
        help="audio backend (default: alsa on Linux, sounddevice elsewhere)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="audio device: ALSA name (default: hw:Gen,0), or one sounddevice index/name used for both input and output",
    )
    p.add_argument(
        "--in-device",
        default=None,
        help="sounddevice input device, index or name (overrides --device)",
    )
    p.add_argument(
        "--out-device",
        default=None,
        help="sounddevice output device, index or name (overrides --device)",
    )
    p.add_argument(
        "--blocksize", type=int, default=4096, help="ALSA period size (default: 4096)"
    )
    p.add_argument(
        "--in-channels",
        type=int,
        default=10,
        help="ALSA capture channel count (default: 10)",
    )
    p.add_argument(
        "--out-channels",
        type=int,
        default=10,
        help="ALSA playback channel count (default: 10)",
    )
    p.add_argument(
        "--list-devices", action="store_true", help="list the audio devices and exit"
    )
    p.add_argument(
        "--analyze",
        type=Path,
        help="re-analyze an existing results folder instead of measuring",
    )
    p.add_argument(
        "--no-show", action="store_true", help="save the plot without opening a window"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        if args.backend == "alsa":
            import alsaaudio

            print("ALSA playback:", alsaaudio.pcms(alsaaudio.PCM_PLAYBACK))
            print("ALSA capture: ", alsaaudio.pcms(alsaaudio.PCM_CAPTURE))
        else:
            import sounddevice as sd

            print(sd.query_devices())
        return

    if args.analyze:
        out_dir = args.analyze
        with open(out_dir / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["nperseg"] = args.nperseg
        meta["show"] = not args.no_show
        rec, _ = sf.read(out_dir / "recording.wav", dtype="float32")
        sweep, _ = sf.read(out_dir / "sweep.wav", dtype="float32")
        analyze(rec, sweep, meta, out_dir)
        return

    if not 0 < args.amplitude <= 1:
        raise SystemExit("--amplitude must be in (0, 1], 1.0 = full scale")
    if max(args.f_start, args.f_end) > args.fs / 2:
        raise SystemExit(f"sweep boundary above Nyquist ({args.fs / 2:g} Hz)")

    sweep = make_sweep(
        args.f_start,
        args.f_end,
        args.duration,
        args.fs,
        args.amplitude,
        args.method,
        args.fade,
    )
    in_chs = [args.ref_ch, args.mic_ch]
    print(
        f"playing {args.f_start:g}-{args.f_end:g} Hz {args.method} sweep, {args.duration:g} ms on out ch{args.out_ch}, "
        f"recording ch{args.ref_ch} (ref) and ch{args.mic_ch} (mic) via {args.backend}"
    )

    if args.backend == "alsa":
        rec = playrec_alsa(
            sweep,
            args.fs,
            args.out_ch,
            in_chs,
            args.device or "hw:Gen,0",
            args.pre,
            args.post,
            args.blocksize,
            args.in_channels,
            args.out_channels,
        )
    else:
        in_device = args.in_device if args.in_device is not None else args.device
        out_device = args.out_device if args.out_device is not None else args.device
        rec = playrec_sounddevice(
            sweep,
            args.fs,
            args.out_ch,
            in_chs,
            in_device,
            out_device,
            args.pre,
            args.post,
        )

    peak = np.max(np.abs(rec), axis=0)
    print(
        f"recorded {len(rec) / args.fs:.2f} s, peak ref {peak[0]:.3f}, peak mic {peak[1]:.3f} (FS = 1)"
    )
    if np.any(peak >= 0.999):
        print("WARNING: clipping detected, lower --amplitude or the input gain")
    if peak[0] < 1e-3:
        print(
            f"WARNING: almost no signal on ch{args.ref_ch}: check that output ch{args.out_ch} is routed "
            "to the speaker (Scarlett mixer/routing) and the speaker/amp is on"
        )

    out_dir = make_test_dir(args.message)
    sf.write(out_dir / "recording.wav", rec, args.fs, subtype="FLOAT")
    sf.write(out_dir / "sweep.wav", sweep, args.fs, subtype="FLOAT")

    meta = {
        "message": out_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "device": args.device,
        "in_device": args.in_device,
        "out_device": args.out_device,
        "fs": args.fs,
        "ref_ch": args.ref_ch,
        "mic_ch": args.mic_ch,
        "out_ch": args.out_ch,
        "f_start": args.f_start,
        "f_end": args.f_end,
        "duration_ms": args.duration,
        "method": args.method,
        "amplitude": args.amplitude,
        "fade": args.fade,
        "pre_s": args.pre,
        "post_s": args.post,
        "nperseg": args.nperseg,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    meta["show"] = not args.no_show
    analyze(rec, sweep, meta, out_dir)


if __name__ == "__main__":
    main()
