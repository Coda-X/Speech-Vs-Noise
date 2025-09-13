import os, csv, numpy as np, librosa, librosa.display, matplotlib.pyplot as plt

SR = 22050

def load_first_mistake(csv_path):
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            return row 
    return None

def pick_reference_example(dataset_npz, desired_label):
    d = np.load(dataset_npz, allow_pickle=True)
    y, files = d["y"].astype(int), d["files"]
    for i, lab in enumerate(y):
        if lab == desired_label:
            return files[i]
    return None

def wavespec(ax_wav, ax_mel, wav_path, title):
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    t = np.arange(len(y))/sr
    ax_wav.plot(t, y)
    ax_wav.set_title(title + " — Waveform")
    ax_wav.set_xlabel("Time (s)")
    ax_wav.set_ylabel("Amplitude")


    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr, ax=ax_mel)
    ax_mel.set_title(title + " — Mel Spectrogram")
    plt.colorbar(img, ax=ax_mel, format="%+0.0f dB")

def main():
    mis_csv = "features/analysis/day8_misclassified.csv"
    if not os.path.exists(mis_csv):
        raise FileNotFoundError("Run day8_analyze_errors.py first to create misclassified.csv")

    row = load_first_mistake(mis_csv)
    if row is None:
        print("No misclassifications found — great job! Add tougher clips or skip this scene.")
        return

    wrong_path = row["file"]
    true_label = int(row["true_label"])
    pred_label = int(row["pred_label"])


    ref_path = pick_reference_example("features/dataset.npz", true_label)
    if ref_path is None:
        print("Couldn't find a reference example for label", true_label)
        return

    print("Misclassified example:", wrong_path, " true=", true_label, " pred=", pred_label)
    print("Reference example:", ref_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    wavespec(axes[0,0], axes[1,0], wrong_path, f"Misclassified (true={true_label}, pred={pred_label})")
    wavespec(axes[0,1], axes[1,1], ref_path,  f"Reference (true={true_label})")
    plt.tight_layout()
    out = "features/plots/day8_wave_mel_compare.png"
    plt.savefig(out, dpi=200)
    plt.show()
    print("Saved:", out)

if __name__ == "__main__":
    main()
