import os, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

DATA_PATH = "features/dataset.npz"
MODEL_PATH = "features/models/day6_best.joblib"
PLOT_DIR = "features/plots"
RNG = 42

Xy = np.load(DATA_PATH, allow_pickle=True)
X, y = Xy["X"], Xy["y"].astype(int)
os.makedirs(PLOT_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=y)
model.fit(Xtr, ytr)
yhat = model.predict(Xte)

acc = accuracy_score(yte, yhat)
print(f"Test Accuracy: {acc:.3f}")

cm = confusion_matrix(yte, yhat, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=["Noise","Speech"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Day 7")
plt.tight_layout()

out_path = os.path.join(PLOT_DIR, "day7_confusion_matrix.png")
plt.savefig(out_path, dpi=200)
plt.show()
print("Saved:", out_path)
