# day9_compare_old_vs_cnn.py
import os, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

DAY6_FEATS = "features/dataset.npz"          # 78-d MFCC stats (Day 6)
DAY6_MODEL = "features/models/day6_best.joblib"
CNN_DATA   = "features/cnn_dataset.npz"      # spectrogram images
CNN_MODEL  = "features/models/day9_cnn_best.h5"
VAL_LIST   = "features/analysis/day9_val_files.txt"
PLOT_DIR   = "features/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load files/labels
d6 = np.load(DAY6_FEATS, allow_pickle=True)
X6_all, y_all, files_all = d6["X"], d6["y"].astype(int), d6["files"]

d9 = np.load(CNN_DATA, allow_pickle=True)
X9_all, y9_all, files9_all = d9["X"], d9["y"].astype(int), d9["files"]

# val files chosen in day9_cnn_train
with open(VAL_LIST, "r") as f:
    val_files = [ln.strip() for ln in f if ln.strip()]

# align indices for both representations
idx9 = [np.where(files9_all == vf)[0][0] for vf in val_files if vf in set(files9_all)]
idx6 = [np.where(files_all   == vf)[0][0] for vf in val_files if vf in set(files_all)]
# ensure same count
common = min(len(idx9), len(idx6))
idx9, idx6 = idx9[:common], idx6[:common]

X9, y9 = X9_all[idx9], y9_all[idx9]
X6, y6 = X6_all[idx6], y_all[idx6]

# Day 6 model
day6 = joblib.load(DAY6_MODEL)
y6_hat = day6.predict(X6)
acc6 = accuracy_score(y6, y6_hat)

# CNN model
import tensorflow as tf
cnn = tf.keras.models.load_model(CNN_MODEL)
y9_hat = (cnn.predict(X9) > 0.5).astype(int).ravel()
acc9 = accuracy_score(y9, y9_hat)

print(f"Day 6 (classic) accuracy on Day 9 val set: {acc6:.3f}")
print(f"Day 9 (CNN)     accuracy on Day 9 val set: {acc9:.3f}")

# Simple bar plot
plt.figure(figsize=(5,4))
plt.bar(["Old Model", "CNN"], [acc6, acc9])
plt.ylim(0,1); plt.ylabel("Accuracy")
plt.title("Day 6 vs Day 9 (CNN) on same val set")
outp = os.path.join(PLOT_DIR, "day9_model_compare.png")
plt.tight_layout(); plt.savefig(outp, dpi=200); plt.show()
print("Saved:", outp)
