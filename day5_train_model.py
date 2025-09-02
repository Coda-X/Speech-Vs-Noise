import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import joblib

DATA_PATH = "features/dataset.npz"     
OUT_DIR   = "features/models"          
MODEL_CHOICE = "logreg"                
TEST_SIZE = 0.3
RANDOM_STATE = 42

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run mfcc_extract.py first.")
    d = np.load(path, allow_pickle=True)
    X, y, files = d["X"], d["y"].astype(int), d["files"]
    return X, y, files

def build_model(name: str):
    if name == "logreg":
       
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elif name == "rf":
   
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    else:
        raise ValueError("MODEL_CHOICE must be 'logreg' or 'rf'.")

def safe_split(X, y):
    vals, cnts = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(vals, cnts)))


    if len(vals) == 2 and cnts.min() >= 2:
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

  
    if len(vals) == 2 and cnts.min() == 1 and len(y) >= 4:
        
        idx0 = np.where(y == vals[0])[0]
        idx1 = np.where(y == vals[1])[0]
        train_idx = np.array([idx0[0], idx1[0]])
        test_idx  = np.setdiff1d(np.arange(len(y)), train_idx)
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

  
    print("⚠️  Not enough balanced samples for a proper split. Using all data for both train and test (demo).")
    return X, X, y, y

def save_outputs(model, acc, report, cm):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUT_DIR, f"day5_{MODEL_CHOICE}.joblib")
    joblib.dump(model, model_path)
    
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Model: {MODEL_CHOICE}\nAccuracy: {acc:.4f}\n\n{report}\n")
        f.write(f"\nConfusion matrix (rows=true [0:Noise,1:Speech], cols=pred):\n{cm}\n")
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {os.path.join(OUT_DIR, 'metrics.txt')}")

def main():
    X, y, files = load_dataset(DATA_PATH)
    print(f"Dataset: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = safe_split(X, y)

    model = build_model(MODEL_CHOICE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

   
    target_names = ["Noise (0)", "Speech (1)"]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print(report)

   
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Noise", "Speech"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix – Day 5 Model")
    plt.tight_layout()
    plt.show()

    save_outputs(model, acc, report, cm)

if __name__ == "__main__":
    main()
