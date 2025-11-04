# day11_make_mix.py
from pathlib import Path
import numpy as np
import soundfile as sf

SR = 22050

def rms(x): 
    return np.sqrt(np.mean(x**2) + 1e-12)

def mix_at_snr(speech_path: Path, noise_path: Path, out_path: Path, snr_db=0):
    s, sr1 = sf.read(speech_path, dtype="float32", always_2d=False)
    n, sr2 = sf.read(noise_path,  dtype="float32", always_2d=False)
    assert sr1 == sr2 == SR, "Use processed 3s files from day11_prepare_real.py"

    L = min(len(s), len(n))
    s, n = s[:L], n[:L]

    s_r = rms(s); n_r = rms(n)
    # set noise level so that (speech RMS / noise RMS) == 10^(snr_db/20)
    target_n_r = s_r / (10**(snr_db/20))
    if n_r > 0:
        n = n * (target_n_r / n_r)

    y = s + n
    # avoid clipping
    peak = np.max(np.abs(y))
    if peak > 0.99: 
        y = 0.99 * y / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, SR, subtype="PCM_16")
    print(f"Saved: {out_path}  (SNR={snr_db} dB)")

if __name__ == "__main__":
    # pick any two processed files you created
    speech = Path("data/real_test/processed/hello_raw_3s.wav")
    noise  = Path("data/real_test/processed/fan_raw_3s.wav")
    out    = Path("data/real_test/processed/mix_0dB.wav")

    mix_at_snr(speech, noise, out, snr_db=0)   # equal levels (hard)
    # You can also try easier cases:
    # mix_at_snr(speech, noise, Path("data/real_test/processed/mix_+6dB.wav"), snr_db=+6)
    # mix_at_snr(speech, noise, Path("data/real_test/processed/mix_-6dB.wav"), snr_db=-6)
