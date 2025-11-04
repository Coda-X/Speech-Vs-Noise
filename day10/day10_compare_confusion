# day10_compare_confusion.py
import os, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths
D6_FEATS = "features/dataset.npz"
D6_MODEL = "features/models/day6_best.joblib"
D9_FEATS = "features/cnn_dataset.npz"
D9_MODEL = "features/models/day9_cnn_best.h5"
VAL_LIST = "features/analysis/day9_val_files.txt"
PLOT_DIR = "features/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_align():
    d6 = np.load(D6_FEATS, allow_pickle=True)
    X6, y6, f6 = d6["X"], d6["y"].astype(int), np.array(list(map(str, d6["files"])), dtype=object)
    d9 = np.load(D9_FEATS, allow_pickle=True)
    X9, y9, f9 = d9["X"], d9["y"].astype(int), np.array(list(map(str, d9["files"])), dtype=object)

    # load validation file list
    if not os.path.exists(VAL_LIST):
        raise FileNotFoundError("VAL list not found. Run day10_compare_accuracy.py first (it will create one if needed).")
    with open(VAL_LIST, "r") as f:
        val_files = [ln.strip() for ln in f if ln.strip()]

    map6 = {fp:i for i, fp in enumerate(f6)}
    map9 = {fp:i for i, fp in enumerate(f9)}
    idx6, idx9 = [], []
    for fp in val_files:
        if fp in map6 and fp in map9:
            idx6.append(map6[fp]); idx9.append(map9[fp])

    X6v, yv6 = X6[idx6], y6[idx6]
    X9v, yv9 = X9[idx9], y9[idx9]
    assert np.all(yv6 == yv9), "Label mismatch between datasets."
    return X6v, X9v, yv6

def main():
    X6v, X9v, yv = load_and_align()
    old_model = joblib.load(D6_MODEL)

    import tensorflow as tf
    cnn = tf.keras.models.load_model(D9_MODEL)

    y_old = old_model.predict(X6v)
    y_cnn = (cnn.predict(X9v) > 0.5).astype(int).ravel()

    cm_old = confusion_matrix(yv, y_old, labels=[0,1])
    cm_cnn = confusion_matrix(yv, y_cnn, labels=[0,1])

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    ConfusionMatrixDisplay(cm_old, display_labels=["Noise","Speech"]).plot(cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Old Model – Confusion Matrix")

    ConfusionMatrixDisplay(cm_cnn, display_labels=["Noise","Speech"]).plot(cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title("CNN – Confusion Matrix")

    fig.suptitle("Day 10 – Old vs CNN (Same Validation Set)")
    plt.tight_layout()
    outp = os.path.join(PLOT_DIR, "day10_confusion_compare.png")
    plt.savefig(outp, dpi=200); plt.show()
    print("Saved:", outp)

if __name__ == "__main__":
    main()
