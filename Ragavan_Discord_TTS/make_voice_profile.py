# make_voice_profile.py
import argparse
import json
import os
from pathlib import Path
import wave


def is_valid_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            fr = wf.getframerate()
            n_frames = wf.getnframes()
            dur = n_frames / float(fr) if fr else 0.0
            # Basic sanity checks (XTTS handles most SRs, but keep clips not-too-short)
            return n_channels in (1, 2) and dur >= 2.0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Create a voice profile from WAVs.")
    parser.add_argument("--name", required=True, help="Profile name (e.g., 'alice')")
    parser.add_argument("--src", required=True, help="Folder with WAV files")
    parser.add_argument("--dst", default="voices", help="Where to store the profile")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")

    wavs = sorted([p for p in src.rglob("*.wav") if is_valid_wav(p)])
    if not wavs:
        raise SystemExit("No valid WAV files found. Ensure clean speech WAVs (≥2s).")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    profile_path = dst / f"{args.name}.json"

    data = {
        "name": args.name,
        "samples": [str(p.resolve()) for p in wavs],
        "notes": "Auto-generated profile. Provide clean, solo speech for best results.",
    }

    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved voice profile: {profile_path}")
    print(f"Samples: {len(wavs)}")


if __name__ == "__main__":
    main()
