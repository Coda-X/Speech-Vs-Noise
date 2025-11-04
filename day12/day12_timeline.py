# day12_timeline.py
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

SR = 22050
WIN_S  = 0.50        # 0.50s analysis window
HOP_S  = 0.25        # 0.25s hop (50% overlap)
WIN_N  = int(WIN_S * SR)
HOP_N  = int(HOP_S * SR)

DATASET_NPZ   = Path("features/dataset.npz")
CLASSIC_MODEL = Path("features/models/day5_logreg.joblib")  # <- your filename
PLOTS_DIR     = Path("features/plots")
ANALYSIS_DIR  = Path("features/analysis")

def mfcc_78_vector(y, sr):
    M  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    D1 = librosa.feature.delta(M)
    D2 = librosa.feature.delta(M, order=2)
    def stats(X):
        mu = np.mean(X, axis=1)
        sd = np.std(X, axis=1)
        return np.concatenate([mu, sd])
    v = np.concatenate([stats(M), stats(D1), stats(D2)])  # 78
    return v.astype(np.float32)

def load_scaler():
    d = np.load(DATASET_NPZ, allow_pickle=True)
    X = d["X"]  # (N, 78)
    sc = StandardScaler().fit(X)
    return sc

def framewise_predict(x, sr, model, scaler):
    probs, t_starts, t_ends = [], [], []
    for start in range(0, max(1, len(x) - WIN_N + 1), HOP_N):
        seg = x[start:start+WIN_N]
        if len(seg) < WIN_N:  # pad last frame
            seg = np.pad(seg, (0, WIN_N - len(seg)))
        v = mfcc_78_vector(seg, sr).reshape(1, -1)
        v = scaler.transform(v)
        p = float(model.predict_proba(v)[0, 1]) if hasattr(model, "predict_proba") else 0.5
        t0 = start / sr
        t1 = (start + WIN_N) / sr
        probs.append(p); t_starts.append(t0); t_ends.append(t1)
    return np.array(t_starts), np.array(t_ends), np.array(probs)

def plot_timeline(wave, t, ts, te, ps, out_png):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wave, linewidth=0.8, alpha=0.5, label="Waveform")
    centers = 0.5*(ts+te)
    ax.plot(centers, ps, linewidth=2.0, label="Prob(Speech)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Day 12 – Speech Probability Timeline")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
    print("Saved plot →", out_png)

def main():
    import sys
    in_wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real_test/processed/mix_0dB.wav")
    assert in_wav.exists(), f"No such file: {in_wav}"

    y, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR); sr = SR

    model  = joblib.load(CLASSIC_MODEL)
    scaler = load_scaler()

    ts, te, ps = framewise_predict(y, sr, model, scaler)  # arrays
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ANALYSIS_DIR / "day12_timeline.csv"
    np.savetxt(out_csv,
               np.c_[ts, te, ps],
               delimiter=",",
               header="t_start,t_end,prob_speech",
               comments="")
    print("Saved timeline →", out_csv)

    t = np.arange(len(y))/sr
    out_png = PLOTS_DIR / "day12_timeline.png"
    plot_timeline(y/np.max(np.abs(y)+1e-9), t, ts, te, ps, out_png)

if __name__ == "__main__":
    main()
