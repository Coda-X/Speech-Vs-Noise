import os, glob, numpy as np
import librosa


SR = 22050
CLIP_SECONDS = 3.0      
N_MFCC = 13
N_FFT = 1024              
HOP_LENGTH = 512          
IN_DIRS = [
    ("data/processed/speech", "speech"),
    ("data/processed/noise", "noise"),
]
OUT_DIR = "features"

os.makedirs(OUT_DIR, exist_ok=True)

def mfcc_matrix(y, sr):
    M = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    d1 = librosa.feature.delta(M)
    d2 = librosa.feature.delta(M, order=2)
    return M, d1, d2

def time_stats(M):
    mu = M.mean(axis=1)
    sd = M.std(axis=1)
    return np.concatenate([mu, sd])  

def process_all():
    X = []
    y = []
    files = []
    label_map = {"speech": 1, "noise": 0}

    for in_root, label_name in IN_DIRS:
        label = label_map[label_name]
        paths = sorted(glob.glob(os.path.join(in_root, "*.wav")))
        for p in paths:
            audio, sr = librosa.load(p, sr=SR, mono=True)

            M, d1, d2 = mfcc_matrix(audio, sr)
            stem = os.path.splitext(os.path.basename(p))[0]
            np.save(os.path.join(OUT_DIR, f"{stem}_mfcc.npy"), M)

            feat = np.concatenate([time_stats(M), time_stats(d1), time_stats(d2)])
            X.append(feat)
            y.append(label)
            files.append(p)

    X = np.vstack(X) if len(X) else np.zeros((0, N_MFCC*2*3))
    y = np.array(y, dtype=np.int64)
    np.savez(os.path.join(OUT_DIR, "dataset.npz"), X=X, y=y, files=np.array(files))
    print(f"Saved: {OUT_DIR}/dataset.npz  |  X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    process_all()
