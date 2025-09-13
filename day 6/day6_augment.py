import os
import glob
import numpy as np
import librosa
import soundfile as sf

SR = 22050           
CLIP_S = 3.0          
L = int(SR * CLIP_S)  


IN_DIRS = [
    ("data/processed/speech", "speech"),
    ("data/processed/noise", "noise"),
]


OUT_ROOT = "data/processed_aug"


AUGS = [
    ("ts09", lambda y: librosa.effects.time_stretch(y, rate=0.90)),   
    ("ts11", lambda y: librosa.effects.time_stretch(y, rate=1.10)),   
    ("ps+2", lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=+2)),  
    ("ps-2", lambda y: librosa.effects.pitch_shift(y, sr=SR, n_steps=-2)),  
    ("gaus", lambda y: y + 0.005 * np.random.randn(len(y))),          
]

def fix_len_norm(y):
    """Ensure clip is exactly 3 seconds and normalized."""
    y = librosa.util.fix_length(data=y, size=L)  
    return librosa.util.normalize(y)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    count = 0

    for in_dir, label in IN_DIRS:
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)

        for p in glob.glob(os.path.join(in_dir, "*.wav")):
            base = os.path.splitext(os.path.basename(p))[0]
            y, _ = librosa.load(p, sr=SR, mono=True)

            for tag, fn in AUGS:
                try:
                    y2 = fix_len_norm(fn(y))
                    out_path = os.path.join(out_dir, f"{base}__{tag}_3s.wav")
                    sf.write(out_path, y2, SR)
                    count += 1
                except Exception as e:
                    print(f"Skipped {p} {tag} error:", e)

    print(f"Saved {count} augmented files in {OUT_ROOT}/...")

if __name__ == "__main__":
    main()
