# day9_build_cnn_dataset.py
import os, glob, numpy as np, librosa, matplotlib.pyplot as plt

SR = 22050
CLIP_S = 3.0
L = int(SR * CLIP_S)

# Spectrogram params
N_FFT = 1024
HOP = 512
N_MELS = 64
W_FIX = 128   # target time frames (≈3s at hop 512)

IN_DIRS = [
    ("data/processed/speech", "speech"),
    ("data/processed/noise",  "noise"),
    ("data/processed_aug/speech", "speech"),
    ("data/processed_aug/noise",  "noise"),
]

OUT_DATA = "features/cnn_dataset.npz"
PLOT_DIR = "features/plots"
os.makedirs("features", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_fixed(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    # exactly 3s (pad/trim) – use keyword args to avoid librosa API issues
    y = librosa.util.fix_length(data=y, size=L)
    return y, sr

def mel_image(y, sr):
    # log-mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)
    # width fix (pad/crop to W_FIX)
    T = S_db.shape[1]
    if T < W_FIX:
        pad = W_FIX - T
        S_db = np.pad(S_db, ((0,0),(0,pad)), mode="edge")
    elif T > W_FIX:
        S_db = S_db[:, :W_FIX]
    # per-sample min-max to [0,1]
    mn, mx = S_db.min(), S_db.max()
    S01 = (S_db - mn) / (mx - mn + 1e-9)
    return S01.astype(np.float32)  # (N_MELS, W_FIX)

def main():
    X, y, files = [], [], []
    label_map = {"speech":1, "noise":0}

    for root, name in IN_DIRS:
        if not os.path.isdir(root):  # skip missing dirs
            continue
        lab = label_map[name]
        for p in glob.glob(os.path.join(root, "*.wav")):
            try:
                a, sr = load_fixed(p)
                img = mel_image(a, sr)              # (64, 128)
                img = np.expand_dims(img, axis=-1)  # (64, 128, 1)
                X.append(img); y.append(lab); files.append(p)
            except Exception as e:
                print("Skip:", p, "|", e)

    if not X:
        raise RuntimeError("No audio found. Make sure processed/ and/or processed_aug/ have wavs.")

    X = np.stack(X, axis=0)           # (N, 64, 128, 1)
    y = np.array(y, dtype=np.int64)
    files = np.array(files, dtype=object)

    np.savez(OUT_DATA, X=X, y=y, files=files)
    print(f"Saved {OUT_DATA} | X={X.shape}, y={y.shape}")

    # Save one example per class
    from random import randint
    os.makedirs(PLOT_DIR, exist_ok=True)
    ids0 = np.where(y==0)[0]
    ids1 = np.where(y==1)[0]
    for lab, ids, nm in [(0, ids0, "noise"), (1, ids1, "speech")]:
        if len(ids)==0: continue
        k = ids[randint(0, len(ids)-1)]
        plt.figure(figsize=(6,3))
        plt.imshow(X[k][:,:,0], aspect="auto", origin="lower")
        plt.title(f"Example log-mel ({nm})\n{os.path.basename(files[k])}")
        plt.xlabel("Time"); plt.ylabel("Mel bins")
        outp = os.path.join(PLOT_DIR, f"day9_example_{nm}.png")
        plt.tight_layout(); plt.savefig(outp, dpi=200); plt.close()
        print("Saved:", outp)

if __name__ == "__main__":
    main()
