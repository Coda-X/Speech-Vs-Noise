# day11_predict.py
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

# ---- paths you already have from earlier days ----
# ...
DATASET_NPZ  = Path("features/dataset.npz")
CLASSIC_MODEL = Path("features/models/day5_logreg.joblib")   # <<< updated
CNN_MODEL     = Path("features/models/day9_cnn_best.h5")


SR = 22050

# ---------- basic loaders ----------
def load_wav(path: Path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y, SR

# ---------- classic model feature (78 dims) ----------
def mfcc_78_vector(y, sr):
    M  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    D1 = librosa.feature.delta(M)
    D2 = librosa.feature.delta(M, order=2)
    def stats(X):
        mu = np.mean(X, axis=1)
        sd = np.std(X, axis=1)
        return np.concatenate([mu, sd])
    v = np.concatenate([stats(M), stats(D1), stats(D2)])  # 13*2*3 = 78
    return v.astype(np.float32)

def load_scaler_from_dataset():
    dset = np.load(DATASET_NPZ, allow_pickle=True)
    X = dset["X"]
    sc = StandardScaler().fit(X)
    return sc

# ---------- CNN feature (log-mel image) ----------
def logmel_image(y, sr, n_mels=64, n_fft=1024, hop=512, frames=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    # pad/trim frames to fixed width
    if S.shape[1] < frames:
        S = np.pad(S, ((0,0),(0, frames - S.shape[1])), mode="edge")
    else:
        S = S[:, :frames]
    S = (S - S.mean()) / (S.std() + 1e-8)
    return S[..., np.newaxis].astype(np.float32)  # (64, frames, 1)

# ---------- predictors ----------
def predict_classic(wav_path: Path):
    assert CLASSIC_MODEL.exists(), f"Missing model: {CLASSIC_MODEL}"
    assert DATASET_NPZ.exists(), f"Missing dataset for scaler: {DATASET_NPZ}"
    model  = joblib.load(CLASSIC_MODEL)
    scaler = load_scaler_from_dataset()

    y, sr = load_wav(wav_path)
    v78 = mfcc_78_vector(y, sr).reshape(1, -1)
    v78 = scaler.transform(v78)

    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(v78)[0, 1])  # prob of Speech=1
    else:
        # fallback if no predict_proba
        s = float(model.decision_function(v78))
        p = 1 / (1 + np.exp(-s))
    lbl = "Speech" if p >= 0.5 else "Noise"
    return lbl, p

def predict_cnn(wav_path: Path):
    import tensorflow as tf
    assert CNN_MODEL.exists(), f"Missing CNN: {CNN_MODEL}"
    net = tf.keras.models.load_model(CNN_MODEL)
    y, sr = load_wav(wav_path)
    img = logmel_image(y, sr)  # (64, 128, 1)
    p = float(net.predict(img[np.newaxis, ...], verbose=0)[0][0])  # prob Speech
    lbl = "Speech" if p >= 0.5 else "Noise"
    return lbl, p

# ---------- CLI ----------
def main():
    # default to processed folder; allow specific file as arg
    if len(sys.argv) == 2 and Path(sys.argv[1]).exists():
        paths = [Path(sys.argv[1])]
    else:
        folder = Path("data/real_test/processed")
        folder.mkdir(parents=True, exist_ok=True)
        paths = sorted(folder.glob("*.wav"))
        if not paths:
            print(f"No files in {folder}. Run day11_prepare_real.py first.")
            return

    for p in paths:
        print(f"\nFile: {p}")
        try:
            cl_lbl, cl_p = predict_classic(p)
            print(f"  Classic → {cl_lbl:6s}  (prob Speech={cl_p:.2f})")
        except Exception as e:
            print(f"  Classic failed: {e}")

        try:
            cnn_lbl, cnn_p = predict_cnn(p)
            print(f"  CNN     → {cnn_lbl:6s}  (prob Speech={cnn_p:.2f})")
        except Exception as e:
            print(f"  CNN failed: {e}")

if __name__ == "__main__":
    main()
