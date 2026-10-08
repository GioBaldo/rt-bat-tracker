"""
Validation of the tracker against the total-station ground truth.

1. For every wav in tests/validation/audio_files (S0.wav, S5.wav, S5_1.wav, ...)
   runs the tracker in audiofile mode:
       python -m rt_bat_tracker.main --mode audiofile --file <wav> --name <stem> --save
   Results are saved by the tracker in results/<stem> (S0, S5, S5_1, ...).
   The tracker closes by itself when the file is over.

2. Collects every localised point of every session and compares it with the
   ground-truth source position taken from TotalStation/M_GROUNDT.TXT
   (only rows with Code == "S"; PtID N is the source of session SN, so S5 and
   S5_1 are both compared with source 5).

3. Expresses points and ground truth in spherical coordinates centred on mic 0:
     - theta = 0 along the normal of the plane through mics 0, 1, 2
       (n = (m1 - m0) x (m2 - m0), oriented towards the sources)
     - phi = 0 along the direction mic0 -> mic1 (lies in the array plane),
       phi positive counter-clockwise around n
   and writes the 3D, radial and angular errors.

Outputs (in tests/validation):
    validation_points.csv   one row per localised point
    validation_summary.csv  one row per session (median / mean errors)

Usage (from the repo root, with the project environment active):
    python tests/validation/run_validation.py               # run all + collect
    python tests/validation/run_validation.py --only S0 S5  # only some sessions
    python tests/validation/run_validation.py --skip-run    # only collect existing results
    python tests/validation/run_validation.py --keep        # don't rerun sessions already in results/

IMPORTANT: the ground truth is in the total-station frame, so the tracker must
run with the mic layout measured in the same frame (default_layout: gt_array.csv)
and with ROTATION_X = ROTATION_Z = 0 and H_DISPLACEMENT = 0 in config.yaml.
The script checks this and warns otherwise.
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VALIDATION_DIR.parents[1]
AUDIO_DIR = VALIDATION_DIR / "audio_files"
TRACKER_AUDIO_DIR = PROJECT_DIR / "data" / "input_audio_files"  # --file only accepts files here
RESULTS_DIR = PROJECT_DIR / "results"
CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"
GT_PATH = PROJECT_DIR / "TotalStation" / "M_GROUNDT.TXT"

POINTS_CSV = VALIDATION_DIR / "validation_points.csv"
SUMMARY_CSV = VALIDATION_DIR / "validation_summary.csv"

EXPECTED_LAYOUT = "gt_array.csv"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def source_id_from_name(name: str) -> int:
    """S5 -> 5, S5_1 -> 5"""
    m = re.match(r"S(\d+)", name)
    if m is None:
        raise ValueError(f"Cannot get source id from '{name}'")
    return int(m.group(1))


def check_config():
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print("WARNING: omegaconf not available, config not checked")
        return
    cfg = OmegaConf.load(CONFIG_PATH)
    problems = []
    if cfg.get("default_layout") != EXPECTED_LAYOUT:
        problems.append(f"default_layout = {cfg.get('default_layout')} (expected {EXPECTED_LAYOUT})")
    for k in ("ROTATION_X", "ROTATION_Z", "H_DISPLACEMENT"):
        if float(cfg.get(k, 0) or 0) != 0:
            problems.append(f"{k} = {cfg.get(k)} (expected 0)")
    if not cfg.get("SAVE_RESULTS", False):
        problems.append("SAVE_RESULTS = False (--save is passed anyway)")
    if problems:
        print("WARNING: config.yaml may not match the ground-truth frame:")
        for p in problems:
            print("   -", p)
        print()


def load_ground_truth():
    gt = pd.read_csv(GT_PATH, skipinitialspace=True)
    gt["Code"] = gt["Code"].astype(str).str.strip().str.upper()
    mics = gt[gt["Code"] == "MIC"][["East", "North", "Height"]].to_numpy(float)
    src = gt[gt["Code"] == "S"].copy()
    src["PtID"] = src["PtID"].astype(str).str.strip().astype(int)
    sources = {
        int(r.PtID): np.array([r.East, r.North, r.Height], dtype=float)
        for r in src.itertuples()
    }
    return mics, sources


class PolarFrame:
    """Spherical frame centred on mic 0 (see module docstring)."""

    def __init__(self, mics: np.ndarray, sources: dict):
        self.origin = mics[0]
        n = np.cross(mics[1] - mics[0], mics[2] - mics[0])
        n /= np.linalg.norm(n)
        # orient the normal towards the sources (mean direction)
        mean_dir = np.mean([s - self.origin for s in sources.values()], axis=0)
        if np.dot(n, mean_dir) < 0:
            n = -n
        e1 = mics[1] - mics[0]
        e1 -= np.dot(e1, n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.n, self.e1, self.e2 = n, e1, e2

    def to_spherical(self, p: np.ndarray):
        v = np.asarray(p, float) - self.origin
        r = np.linalg.norm(v)
        if r == 0:
            return 0.0, 0.0, 0.0
        theta = np.degrees(np.arccos(np.clip(np.dot(v, self.n) / r, -1, 1)))
        phi = np.degrees(np.arctan2(np.dot(v, self.e2), np.dot(v, self.e1)))
        return r, theta, phi

    def angle_between(self, p: np.ndarray, q: np.ndarray) -> float:
        """Angle (deg) between directions origin->p and origin->q."""
        a = np.asarray(p, float) - self.origin
        b = np.asarray(q, float) - self.origin
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def wrap180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# 1. run the tracker
# --------------------------------------------------------------------------
def wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 60.0


def ensure_in_tracker_audio_dir(wav: Path):
    """cli.py only accepts files in data/input_audio_files."""
    TRACKER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRACKER_AUDIO_DIR / wav.name
    if not dest.exists() or dest.stat().st_size != wav.stat().st_size:
        print(f"  copying {wav.name} to data/input_audio_files")
        shutil.copy2(wav, dest)


def _clear_readonly_and_retry(func, path, *_):
    """rmtree error handler: on Windows files/folders can be read-only
    (or briefly locked by OneDrive / Explorer thumbnails). Clear the flag,
    wait a moment and retry once."""
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        pass
    time.sleep(0.5)
    func(path)


def remove_old_session(session_dir: Path, attempts: int = 3) -> bool:
    """Delete an old session folder. If Windows keeps refusing (locked files),
    rename it to <name>_old_<timestamp> instead so the new run can start.
    Returns False only if both fail."""
    for i in range(attempts):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(session_dir, onexc=_clear_readonly_and_retry)
            else:
                shutil.rmtree(session_dir, onerror=_clear_readonly_and_retry)
            return True
        except OSError as e:
            last_err = e
            time.sleep(1.0 + i)
        if not session_dir.exists():
            return True

    backup = session_dir.with_name(f"{session_dir.name}_old_{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        session_dir.rename(backup)
        print(f"  could not delete {session_dir.name} ({last_err}); moved it to {backup.name}")
        return True
    except OSError as e:
        print(f"  could not delete ({last_err}) nor rename ({e}) {session_dir}")
        return False


def run_tracker(wav: Path, keep: bool) -> bool:
    name = wav.stem
    session_dir = RESULTS_DIR / name

    if session_dir.exists():
        if keep:
            print(f"[{name}] results already present, skipped (--keep)")
            return True
        if (session_dir / "session_manifest.json").exists() or not any(session_dir.iterdir()):
            # old run of the same session: start clean
            if not remove_old_session(session_dir):
                print(
                    f"[{name}] ERROR: cannot remove or rename {session_dir} "
                    "(is a file open, or is OneDrive syncing it?). Skipped."
                )
                return False
        else:
            print(f"[{name}] WARNING: {session_dir} exists and is not a session folder, skipped")
            return False

    ensure_in_tracker_audio_dir(wav)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-m", "rt_bat_tracker.main",
        "--mode", "audiofile",
        "--file", wav.name,
        "--name", name,
        "--save",
    ]
    timeout = wav_duration(wav) * 3 + 60  # file is played in real time

    print(f"[{name}] running tracker ({wav_duration(wav):.1f} s of audio)...")
    t0 = time.time()
    log_path = VALIDATION_DIR / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, cwd=PROJECT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"[{name}] ERROR: timeout after {timeout:.0f} s (log: {log_path})")
        return False

    ok = proc.returncode == 0 and (session_dir / "session_manifest.json").exists()
    status = "done" if ok else f"FAILED (exit {proc.returncode}, log: {log_path})"
    print(f"[{name}] {status} in {time.time() - t0:.1f} s")
    return ok


# --------------------------------------------------------------------------
# 2-3. collect results
# --------------------------------------------------------------------------
def collect(names, frame: PolarFrame, sources: dict) -> pd.DataFrame:
    rows = []
    for name in names:
        session_dir = RESULTS_DIR / name
        manifest_path = session_dir / "session_manifest.json"
        src_id = source_id_from_name(name)
        gt = sources.get(src_id)

        if gt is None:
            print(f"[{name}] WARNING: no ground truth for source {src_id}")
            continue
        if not manifest_path.exists():
            print(f"[{name}] no results found")
            continue

        gt_r, gt_theta, gt_phi = frame.to_spherical(gt)

        with open(manifest_path, encoding="utf-8") as f:
            events = json.load(f).get("events", [])

        for ev in events:
            ev_dir = session_dir / ev["event_folder"]
            pts_path = ev_dir / ev["points_file"]
            if not pts_path.exists():
                continue
            pts = pd.read_csv(pts_path)

            # optional: analytics of the same event, merged by point_idx
            an_path = ev_dir / f"{ev['event_folder']}.analytics.csv"
            analytics = None
            if an_path.exists() and "point_idx" in pts.columns:
                analytics = pd.read_csv(an_path, dtype={"selected_mics_str": str})
                analytics = analytics.drop(
                    columns=[c for c in ("rel_ts", "score") if c in analytics.columns]
                ).set_index("point_idx")

            for i, p in pts.iterrows():
                pos = np.array([p["x"], p["y"], p["z"]], dtype=float)
                if not np.all(np.isfinite(pos)):
                    continue
                r, theta, phi = frame.to_spherical(pos)
                diff = pos - gt
                row = {
                    "file": name,
                    "source_id": src_id,
                    "event": ev["event_folder"],
                    "point_idx": int(p.get("point_idx", i)),
                    "abs_ts": p.get("abs_ts", np.nan),
                    "rel_ts": p.get("rel_ts", np.nan),
                    "score": p.get("score", np.nan),
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2],
                    "err_x": diff[0], "err_y": diff[1], "err_z": diff[2],
                    "err_3d": float(np.linalg.norm(diff)),
                    "r": r, "theta_deg": theta, "phi_deg": phi,
                    "gt_r": gt_r, "gt_theta_deg": gt_theta, "gt_phi_deg": gt_phi,
                    "err_r": r - gt_r,                        # signed: + = too far
                    "err_theta_deg": theta - gt_theta,
                    "err_phi_deg": wrap180(phi - gt_phi),
                    "err_angle_deg": frame.angle_between(pos, gt),  # total angular error
                }
                if analytics is not None and row["point_idx"] in analytics.index:
                    for col, val in analytics.loc[row["point_idx"]].items():
                        row[f"an_{col}"] = val
                rows.append(row)

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("file", sort=False)
    summary = pd.DataFrame({
        "source_id": g["source_id"].first(),
        "n_points": g.size(),
        "gt_r": g["gt_r"].first(),
        "gt_theta_deg": g["gt_theta_deg"].first(),
        "gt_phi_deg": g["gt_phi_deg"].first(),
        "median_err_3d": g["err_3d"].median(),
        "mean_err_3d": g["err_3d"].mean(),
        "median_err_r": g["err_r"].median(),
        "median_abs_err_r": g["err_r"].apply(lambda s: s.abs().median()),
        "median_err_angle_deg": g["err_angle_deg"].median(),
        "mean_err_angle_deg": g["err_angle_deg"].mean(),
        "median_err_theta_deg": g["err_theta_deg"].median(),
        "median_err_phi_deg": g["err_phi_deg"].median(),
        "median_score": g["score"].median(),
    }).reset_index()
    return summary


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="run/collect only these names (e.g. S0 S5 S5_1)")
    parser.add_argument("--skip-run", action="store_true", help="don't run the tracker, only collect results")
    parser.add_argument("--keep", action="store_true", help="don't rerun sessions already present in results/")
    args = parser.parse_args()

    wavs = sorted(AUDIO_DIR.glob("*.wav"), key=lambda p: natural_key(p.stem))
    if args.only:
        wavs = [w for w in wavs if w.stem in set(args.only)]
    if not wavs:
        raise SystemExit(f"No wav files found in {AUDIO_DIR}")
    names = [w.stem for w in wavs]

    mics, sources = load_ground_truth()
    frame = PolarFrame(mics, sources)

    if not args.skip_run:
        check_config()
        failed = [w.stem for w in wavs if not run_tracker(w, args.keep)]
        if failed:
            print(f"\nFailed runs: {', '.join(failed)}\n")

    df = collect(names, frame, sources)
    if df.empty:
        print("No points collected.")
        return

    df.to_csv(POINTS_CSV, index=False)
    summary = summarise(df)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\n{len(df)} points from {df['file'].nunique()} sessions")
    print(f"  median 3D error:      {df['err_3d'].median():.3f} m")
    print(f"  median |radial err|:  {df['err_r'].abs().median():.3f} m")
    print(f"  median angular error: {df['err_angle_deg'].median():.2f} deg")
    print(f"\nSaved {POINTS_CSV.name} and {SUMMARY_CSV.name} in {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
