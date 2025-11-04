# day12_apply_gate.py
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd

SR = 22050
ATTEN_DB   = -12.0             # gain for "Noise" frames
THRESH     = 0.6               # prob(Speech) threshold
SMOOTH_WIN = 3                 # simple smoothing across frames

def db_to_lin(db): return 10**(db/20)

def smooth(x, k=3):
    if k <= 1: return x
    k = int(k)
    pad = (k-1)//2
    xp = np.pad(x, (pad,pad), mode="edge")
    w  = np.ones(k)/k
    return np.convolve(xp, w, mode="valid")

def make_sample_gain(len_samples, sr, t_start, t_end, probs, thresh, atten_db):
    # frame labels → per-sample gain
    noise_gain = db_to_lin(atten_db)
    centers = 0.5*(t_start + t_end)
    prob_s   = smooth(probs, k=SMOOTH_WIN)  # a little smoothing
    labels   = (prob_s >= thresh).astype(np.float32)  # 1 = speech, 0 = noise
    # upsample to sample grid by nearest-center
    times = np.arange(len_samples)/sr
    idx   = np.searchsorted(centers, times, side="left")
    idx   = np.clip(idx, 0, len(labels)-1)
    frame_label = labels[idx]
    gain = np.where(frame_label > 0.5, 1.0, noise_gain).astype(np.float32)
    return gain

def main():
    import sys
    in_wav  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real_test/processed/mix_0dB.wav")
    timeline_csv = Path("features/analysis/day12_timeline.csv")
    assert in_wav.exists(), f"No input wav: {in_wav}"
    assert timeline_csv.exists(), f"Missing timeline: {timeline_csv} (run day12_timeline.py first)"

    y, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    assert sr == SR, f"Expected {SR} Hz audio; got {sr}"

    df = pd.read_csv(timeline_csv)
    t0 = df["t_start"].values
    t1 = df["t_end"].values
    ps = df["prob_speech"].values

    gain = make_sample_gain(len(y), sr, t0, t1, ps, THRESH, ATTEN_DB)
    y_out = y * gain
    # prevent clipping
    peak = np.max(np.abs(y_out))
    if peak > 0.99: y_out = 0.99 * y_out / peak

    out_dir = Path("outputs/day12")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"enhanced_thresh{THRESH}_att{int(ATTEN_DB)}dB.wav"
    sf.write(out_wav, y_out, sr, subtype="PCM_16")

    print(f"Saved enhanced audio → {out_wav}")
    print(f"Params: threshold={THRESH}, noise_gain={ATTEN_DB} dB, smooth_win={SMOOTH_WIN}")

if __name__ == "__main__":
    main()
