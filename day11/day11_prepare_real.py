# day11_prepare_real.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import sys

SR = 22050
TARGET_S = 3.0
TARGET_N = int(SR * TARGET_S)

# Default locations (can be overridden via CLI)
DEFAULT_RAW_DIR = Path("data/real_test/raw")
OUT_DIR = Path("data/real_test/processed")

def to_mono(y):
    return y if y.ndim == 1 else y.mean(axis=1)

def trim_pad(y, n):
    if len(y) > n: return y[:n]
    if len(y) < n: return np.pad(y, (0, n - len(y)))
    return y

def process_one(src: Path, dst: Path):
    y, sr = sf.read(src, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = to_mono(y)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = trim_pad(y, TARGET_N)
    peak = np.max(np.abs(y)) if y.size else 0
    if peak > 0:
        y = 0.95 * (y / peak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, SR, subtype="PCM_16")
    print(f"✓ Saved 3s: {dst}")

def main():
    # Allow an optional input folder:  python day11_prepare_real.py path/to/folder
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW_DIR

    # Create folders if missing
    raw_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in raw_dir.glob("*.wav"))
    print(f"[IN ] {raw_dir}  |  found {len(files)} wav(s)")
    print(f"[OUT] {OUT_DIR}")

    if not files:
        print("\nNo WAVs found.")
        print("Drop your recordings here and re-run:")
        print(f"  {raw_dir}\n")
        print("Examples:")
        print("  hello_raw.wav, fan_raw.wav, street_raw.wav, mix_raw.wav\n")
        return

    for p in files:
        out_name = p.stem.lower().replace(" ", "_") + "_3s.wav"
        out_path = OUT_DIR / out_name
        try:
            process_one(p, out_path)
        except Exception as e:
            print(f"✗ Skipped {p.name}: {e}")

    print("\nDONE. Processed files are in:", OUT_DIR)

if __name__ == "__main__":
    main()
