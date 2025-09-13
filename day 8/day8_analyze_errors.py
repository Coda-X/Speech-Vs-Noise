import os, csv, shutil, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

DATA_PATH   = "features/dataset.npz"
MODEL_PATH  = "features/models/day6_best.joblib"
ANALY_DIR   = "features/analysis"
FAIL_DIR    = "data/failures"
PLOTS_DIR   = "features/plots"
RNG         = 42
LABEL_NAME  = {0: "noise", 1: "speech"}

def ensure_dirs():
    os.makedirs(ANALY_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    d = np.load(DATA_PATH, allow_pickle=True)
    X, y, files = d["X"], d["y"].astype(int), d["files"]
    return X, y, files

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run tuning (Day 6) first.")
    return joblib.load(MODEL_PATH)

def copy_failure(src_path, true_lab, pred_lab):
    sub = f"true_{LABEL_NAME[true_lab]}__pred_{LABEL_NAME[pred_lab]}"
    out_dir = os.path.join(FAIL_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(src_path)
    dst  = os.path.join(out_dir, base)
    if os.path.abspath(src_path) != os.path.abspath(dst):
        try:
            shutil.copy2(src_path, dst)
        except Exception as e:
            print("copy failed:", src_path, "->", dst, "|", e)

def main():
    ensure_dirs()
    X, y, files = load_data()
    model = load_model()

    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(
        X, y, files, test_size=0.3, random_state=RNG, stratify=y
    )

    model.fit(Xtr, ytr)


    yhat = model.predict(Xte)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xte)   
    else:
        probs = None


    acc = accuracy_score(yte, yhat)
    print(f"Test Accuracy: {acc:.3f}")
    print(classification_report(yte, yhat, target_names=["Noise (0)", "Speech (1)"], digits=3))


    cm = confusion_matrix(yte, yhat, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=["Noise","Speech"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix – Day 8")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "day8_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.show()
    print("Saved:", cm_path)


    pred_csv = os.path.join(ANALY_DIR, "day8_predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "pred_label", "prob_noise", "prob_speech"])
        for i, fp in enumerate(fte):
            p0 = float(probs[i,0]) if probs is not None else ""
            p1 = float(probs[i,1]) if probs is not None else ""
            w.writerow([fp, int(yte[i]), int(yhat[i]), p0, p1])
    print("Saved:", pred_csv)


    mis_csv = os.path.join(ANALY_DIR, "day8_misclassified.csv")
    mis_count = 0
    with open(mis_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_label", "pred_label", "prob_noise", "prob_speech"])
        for i, fp in enumerate(fte):
            if yhat[i] != yte[i]:
                mis_count += 1
                p0 = float(probs[i,0]) if probs is not None else ""
                p1 = float(probs[i,1]) if probs is not None else ""
                w.writerow([fp, int(yte[i]), int(yhat[i]), p0, p1])
                copy_failure(fp, yte[i], yhat[i])

    print(f"Misclassified samples: {mis_count}")
    if mis_count > 0:
        print("Copied misclassified files to:", FAIL_DIR)

if __name__ == "__main__":
    main()
