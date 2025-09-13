import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, validation_curve
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = "features/dataset.npz"
BEST_PATH = "features/models/day6_best.joblib"
RNG = 42

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run day6_build_dataset.py first.")
    d = np.load(DATA_PATH, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y size mismatch: X={X.shape}, y={y.shape}")
    if len(np.unique(y)) < 2 or len(y) < 2:
        raise ValueError("Dataset must contain at least 2 samples and 2 classes to evaluate.")
    return X, y

def load_or_default_model():
    if os.path.exists(BEST_PATH):
        print("Loading tuned model:", BEST_PATH)
        return joblib.load(BEST_PATH), "tuned"
    print("No tuned model found → using default Logistic Regression.")
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RNG)), "default"

def main():
    X, y = load_dataset()


    strat = y if np.min(np.bincount(y)) >= 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=RNG, stratify=strat)

    model, tag = load_or_default_model()
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    acc = accuracy_score(yte, yhat)
    print(f"Test Accuracy ({tag}): {acc:.3f}")
    print(classification_report(yte, yhat, target_names=["Noise (0)", "Speech (1)"], digits=3))

    cm = confusion_matrix(yte, yhat, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Noise", "Speech"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix – Day 6")
    plt.tight_layout()
    plt.show()


    if np.min(np.bincount(y)) >= 5:
        print("Building validation curve for C (Logistic Regression)…")
        cv = StratifiedKFold(
            n_splits=min(5, np.min(np.bincount(y))),
            shuffle=True,
            random_state=RNG
        )
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RNG))
        C_vals = np.logspace(-2, 2, 7)


        train_scores, val_scores = validation_curve(
            estimator=pipe,
            X=X,
            y=y,
            param_name="logisticregression__C",
            param_range=C_vals,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,   
        )

        plt.figure(figsize=(7, 4))
        plt.semilogx(C_vals, train_scores.mean(axis=1), marker="o", label="train")
        plt.semilogx(C_vals, val_scores.mean(axis=1), marker="o", label="cv")
        plt.xlabel("C (Logistic Regression)")
        plt.ylabel("Accuracy")
        plt.title("Validation Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough samples per class for a stable validation curve; skipping.")

if __name__ == "__main__":
    main()
