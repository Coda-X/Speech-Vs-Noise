import os, glob, numpy as np, librosa, soundfile as sf
def preprocess_wav(in_path, out_path, target_sr=22050, clip_s=3.0, trim_db=30):
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    y=librosa.util.normalize(y)
    need = int(clip_s*target_sr)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    else:
        y = y[:need]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, target_sr)

def batch_process():
    for p in glob.glob("data/noise/*.wav"):
        name = os.path.splitext(os.path.basename(p))[0]
        preprocess_wav(p, f"data/processed/noise/{name}_3s.wav")
    for p in glob.glob("data/speech/*.wav"):
        name = os.path.splitext(os.path.basename(p))[0]
        preprocess_wav(p, f"data/processed/speech/{name}_3s.wav")

if __name__ == "__main__":
    batch_process()
    print("Done: processed 3s clips in data/processed/ ...")


   
