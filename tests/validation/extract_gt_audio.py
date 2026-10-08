"""
Extract the audio files of all ground-truth sessions.

Reads every session folder in GT_sessions (S0, S1, ...) and copies its event
wav files into audio_files, named after the session folder:

    first event   -> S5.wav
    second event  -> S5_1.wav
    third event   -> S5_2.wav
    ...

Events are ordered by start_time_adc from session_manifest.json (falls back to
the event number if the manifest is missing). Files are copied as they are,
without re-encoding. Existing files in audio_files with the same name are
overwritten.

Usage (from anywhere):
    python tests/validation/extract_gt_audio.py
"""

import json
import re
import shutil
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = VALIDATION_DIR / "GT_sessions"
OUTPUT_DIR = VALIDATION_DIR / "audio_files"


def natural_key(path: Path):
    """Sort S2 before S10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def ordered_event_wavs(session_dir: Path) -> list[Path]:
    """Return the session's event wav files in chronological order."""
    manifest_path = session_dir / "session_manifest.json"

    if manifest_path.exists():
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            events = json.load(f).get("events", [])
        events.sort(key=lambda e: float(e.get("start_time_adc", 0.0)))
        wavs = [
            session_dir / e["event_folder"] / e["audio_file"]
            for e in events
            if "event_folder" in e and "audio_file" in e
        ]
        missing = [w for w in wavs if not w.exists()]
        for w in missing:
            print(f"  WARNING: {w.relative_to(SESSIONS_DIR)} listed in manifest but not found")
        return [w for w in wavs if w.exists()]

    # no manifest: take Event_N folders sorted by N
    print(f"  WARNING: no manifest in {session_dir.name}, ordering by event number")
    return sorted(session_dir.glob("Event_*/*.wav"), key=lambda w: natural_key(w.parent))


def main():
    if not SESSIONS_DIR.exists():
        raise SystemExit(f"Sessions folder not found: {SESSIONS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = sorted((d for d in SESSIONS_DIR.iterdir() if d.is_dir()), key=natural_key)
    total = 0

    for session_dir in sessions:
        wavs = ordered_event_wavs(session_dir)
        if not wavs:
            print(f"{session_dir.name}: no audio files found, skipped")
            continue

        for i, wav in enumerate(wavs):
            name = session_dir.name if i == 0 else f"{session_dir.name}_{i}"
            dest = OUTPUT_DIR / f"{name}.wav"
            shutil.copy2(wav, dest)
            print(f"{session_dir.name}/{wav.parent.name}/{wav.name} -> {dest.name}")
            total += 1

    print(f"\nDone: {total} files from {len(sessions)} sessions copied to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
