import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = "features/dataset.npz"
PLOT_DIR  = "features/plots"
RNG = 42

def load_dataset():
    d = np.load(DATA_PATH, allow_pickle=True)
    return d["X"], d["y"].astype(int)

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    X, y = load_dataset()
    cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=RNG)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RNG))


    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipe,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        train_sizes=np.linspace(0.2, 1.0, 6),
        shuffle=True,
        random_state=RNG,
    )
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train accuracy")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label="CV accuracy")
    plt.xlabel("Training size (samples)")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (Accuracy vs Training Size)")
    plt.legend()
    plt.tight_layout()
    lc_path = os.path.join(PLOT_DIR, "day7_learning_curve.png")
    plt.savefig(lc_path, dpi=200)
    plt.show()
    print("Saved:", lc_path)


    C_vals = np.logspace(-2, 2, 7)
    tr, va = validation_curve(
        estimator=pipe,
        X=X,
        y=y,
        param_name="logisticregression__C",
        param_range=C_vals,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    plt.figure(figsize=(7.5, 4.5))
    plt.semilogx(C_vals, tr.mean(axis=1), marker="o", label="Train acc")
    plt.semilogx(C_vals, va.mean(axis=1), marker="o", label="CV acc")
    plt.xlabel("C (Logistic Regression)")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve (Model Complexity)")
    plt.legend()
    plt.tight_layout()
    vc_path = os.path.join(PLOT_DIR, "day7_validation_curve.png")
    plt.savefig(vc_path, dpi=200)
    plt.show()
    print("Saved:", vc_path)

if __name__ == "__main__":
    main()
