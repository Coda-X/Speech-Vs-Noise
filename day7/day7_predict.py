import os, numpy as np, librosa, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

SR = 22050
N_MFCC, N_FFT, HOP = 13, 1024, 512
MODEL_PATH = "features/models/day6_best.joblib"

def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    M  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(M)
    d2 = librosa.feature.delta(M, order=2)
    def stats(A): 
        mu, sd = A.mean(axis=1), A.std(axis=1)
        return np.concatenate([mu, sd])
    feat = np.concatenate([stats(M), stats(d1), stats(d2)]) 
    return feat.reshape(1, -1)

def main(paths):
    if not os.path.exists(MODEL_PATH):

        print("No tuned model found; using default LR.")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    else:
        model = joblib.load(MODEL_PATH)

    for p in paths:
        x = extract_features(p)
        pred = model.predict(x)[0]
        proba = getattr(model, "predict_proba", lambda z: None)(x)
        label = "Speech (1)" if pred == 1 else "Noise (0)"
        if proba is not None:
            conf = float(np.max(proba))
            print(f"{os.path.basename(p)} → {label}  (confidence {conf:.2f})")
        else:
            print(f"{os.path.basename(p)} → {label}")

if __name__ == "__main__":
    test_paths = []
    for name in ["data/new_test/hello.wav", "data/new_test/fan.wav"]:
        if os.path.exists(name): test_paths.append(name)
    if not test_paths:
        print("Place test WAVs in data/new_test/, e.g., hello.wav, fan.wav")
    else:
        main(test_paths)
