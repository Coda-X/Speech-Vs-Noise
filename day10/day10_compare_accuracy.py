# day10_compare_accuracy.py
import os, numpy as np, joblib, matplotlib.pyplot as plt

# Paths
D6_FEATS = "features/dataset.npz"              # Day 6 features (78-d)
D6_MODEL = "features/models/day6_best.joblib"  # Day 6 model
D9_FEATS = "features/cnn_dataset.npz"          # Day 9 spectrogram images
D9_MODEL = "features/models/day9_cnn_best.h5"  # Day 9 CNN
VAL_LIST = "features/analysis/day9_val_files.txt"  # preferred split from Day 9
PLOT_DIR = "features/plots"
AN_DIR   = "features/analysis"
RNG = 42

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(AN_DIR, exist_ok=True)

def load_all():
    d6 = np.load(D6_FEATS, allow_pickle=True)
    X6, y6, f6 = d6["X"], d6["y"].astype(int), d6["files"]

    d9 = np.load(D9_FEATS, allow_pickle=True)
    X9, y9, f9 = d9["X"], d9["y"].astype(int), d9["files"]
    return (X6, y6, f6), (X9, y9, f9)

def build_index(files):
    # map file path -> index
    return {str(fp): i for i, fp in enumerate(files)}

def choose_val_files(f6, f9):
    # Prefer Day 9 val list if present
    if os.path.exists(VAL_LIST):
        with open(VAL_LIST, "r") as f:
            lst = [ln.strip() for ln in f if ln.strip()]
        return lst

    # Else: create a shared validation split from intersection of files
    set6, set9 = set(map(str, f6)), set(map(str, f9))
    common = np.array(sorted(set6.intersection(set9)), dtype=object)
    if common.size < 10:
        raise RuntimeError("Too few common files to form a validation set. Rebuild datasets first.")
    rng = np.random.default_rng(RNG)
    rng.shuffle(common)
    k = max( int(0.2 * len(common)), 10)  # ~20% or at least 10 files
    val = common[:k].tolist()

    # Save for reproducibility
    os.makedirs(os.path.dirname(VAL_LIST), exist_ok=True)
    with open(VAL_LIST, "w") as f:
        for fp in val: f.write(fp + "\n")
    print(f"Saved shared validation list → {VAL_LIST} ({len(val)} files)")
    return val

def align_by_files(val_files, f6, f9):
    idx6 = []
    idx9 = []
    map6, map9 = build_index(f6), build_index(f9)
    for fp in val_files:
        if fp in map6 and fp in map9:
            idx6.append(map6[fp])
            idx9.append(map9[fp])
    if len(idx6) == 0 or len(idx9) == 0:
        raise RuntimeError("No overlapping files between datasets for selected validation list.")
    return np.array(idx6), np.array(idx9)

def proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1]  # prob of class 1 (speech)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        # map to [0,1] via logistic for display consistency
        return 1.0 / (1.0 + np.exp(-z))
    return None

def main():
    (X6, y6, f6), (X9, y9, f9) = load_all()
    # File strings for dict keys
    f6 = np.array(list(map(str, f6)), dtype=object)
    f9 = np.array(list(map(str, f9)), dtype=object)

    val_files = choose_val_files(f6, f9)
    i6, i9 = align_by_files(val_files, f6, f9)

    X6v, yv6 = X6[i6], y6[i6]
    X9v, yv9 = X9[i9], y9[i9]
    assert np.all(yv6 == yv9), "Label mismatch between Day 6 and Day 9 datasets."
    yv = yv6

    # Load models
    old_model = joblib.load(D6_MODEL)

    import tensorflow as tf
    cnn = tf.keras.models.load_model(D9_MODEL)

    # Predictions
    y_old = old_model.predict(X6v)
    y_cnn = (cnn.predict(X9v) > 0.5).astype(int).ravel()

    # Accuracies
    acc_old = (y_old == yv).mean()
    acc_cnn = (y_cnn == yv).mean()
    print(f"Old model accuracy: {acc_old:.3f}  |  CNN accuracy: {acc_cnn:.3f}")

    # Optional: save per-sample CSV for reference
    p_old = proba_or_score(old_model, X6v)
    p_cnn = cnn.predict(X9v).ravel()
    out_csv = os.path.join(AN_DIR, "day10_predictions.csv")
    with open(out_csv, "w") as f:
        f.write("file,true,pred_old,prob_old_speech,pred_cnn,prob_cnn_speech\n")
        for k, fp in enumerate(val_files[:len(yv)]):
            po = "" if p_old is None else float(p_old[k])
            pc = float(p_cnn[k])
            f.write(f"{fp},{int(yv[k])},{int(y_old[k])},{po},{int(y_cnn[k])},{pc}\n")
    print("Saved:", out_csv)

    # Bar chart
    plt.figure(figsize=(5,4))
    plt.bar(["Old Model","CNN"], [acc_old, acc_cnn])
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Old vs CNN – Validation Accuracy")
    outp = os.path.join(PLOT_DIR, "day10_accuracy_compare.png")
    plt.tight_layout(); plt.savefig(outp, dpi=200); plt.show()
    print("Saved:", outp)

if __name__ == "__main__":
    main()
