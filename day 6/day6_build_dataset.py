import os, glob, numpy as np, librosa

SR = 22050
N_MFCC, N_FFT, HOP = 13, 1024, 512
OUT_DIR = "features"


IN_DIRS = [
    ("data/processed/speech", "speech"),
    ("data/processed/noise",  "noise"),
    ("data/processed_aug/speech", "speech"),
    ("data/processed_aug/noise",  "noise"),
]

def mfcc_triplet(y, sr):
    """MFCC + delta + delta-delta matrices."""
    M  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(M)
    d2 = librosa.feature.delta(M, order=2)
    return M, d1, d2

def stats(M):
    """Concatenate mean and std over time axis for one matrix."""
    mu, sd = M.mean(axis=1), M.std(axis=1)
    return np.concatenate([mu, sd])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y, files = [], [], []
    label_map = {"speech": 1, "noise": 0}

    for root, name in IN_DIRS:
        if not os.path.isdir(root):
            continue 
        lab = label_map[name]
        for p in glob.glob(os.path.join(root, "*.wav")):
            try:
                a, sr = librosa.load(p, sr=SR, mono=True)
                M, d1, d2 = mfcc_triplet(a, sr)
                feat = np.concatenate([stats(M), stats(d1), stats(d2)])  
                X.append(feat); y.append(lab); files.append(p)
            except Exception as e:
                print("Skip (feature error):", p, "|", e)

    if X:
        X = np.vstack(X)
        y = np.array(y, dtype=int)
        files = np.array(files, dtype=object)
    else:
        X = np.zeros((0, 78), dtype=float)
        y = np.zeros((0,), dtype=int)
        files = np.array([], dtype=object)

    np.savez(os.path.join(OUT_DIR, "dataset.npz"), X=X, y=y, files=files)


    print(f"Saved features/dataset.npz | X={X.shape}, y={y.shape}")
    if y.size:
        vals, cnts = np.unique(y, return_counts=True)
        print("Class counts:", dict(zip(vals, cnts)))
    else:
        print("Class counts: (empty dataset)")

if __name__ == "__main__":
    main()
