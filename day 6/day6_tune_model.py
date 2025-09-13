import os, numpy as np, joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend

DATA_PATH = "features/dataset.npz"
OUT_DIR   = "features/models"
RNG       = 42

def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run your feature builder first.")
    d = np.load(path, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    return X, y

def choose_cv(y):
    vals, cnts = np.unique(y, return_counts=True)
    k = min(5, cnts.min()) if len(vals) >= 2 else 0
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=RNG) if k >= 2 else None

def full_grid_search(X, y, cv):
   
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RNG))
    lr_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}
    lr_gs = GridSearchCV(lr, lr_grid, cv=cv, n_jobs=1)

    rf = RandomForestClassifier(random_state=RNG)
    rf_grid = {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10], "max_features": ["sqrt", 0.5, 1.0]}
    rf_gs = GridSearchCV(rf, rf_grid, cv=cv, n_jobs=1)

    best_est, best_name, best_score = None, None, -1.0
    with parallel_backend("threading", n_jobs=1):
        for name, gs in (("logreg", lr_gs), ("rf", rf_gs)):
            gs.fit(X, y)
            print(f"{name} best params: {gs.best_params_} | CV mean acc: {gs.best_score_:.3f}")
            if gs.best_score_ > best_score:
                best_est, best_name, best_score = gs.best_estimator_, name, gs.best_score_
    print(f"\nSelected best model: {best_name} (CV acc={best_score:.3f})")
    return best_est

def tiny_data_fallback(X, y):
    print("Not enough per-class samples for CV; using a simple holdout for tuning.")
    strat = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 1 else None
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.4, random_state=RNG, stratify=strat)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=RNG))
    grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}
    gs = GridSearchCV(lr, grid, cv=2, n_jobs=1)
    with parallel_backend("threading", n_jobs=1):
        gs.fit(X_tr, y_tr)
    print("Best (fallback) LR params:", gs.best_params_)
    return gs.best_estimator_

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_dataset()
    print(f"Dataset: X={X.shape}, y={y.shape}")
    print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

    cv = choose_cv(y)
    best = tiny_data_fallback(X, y) if cv is None else full_grid_search(X, y, cv)

    out_path = os.path.join(OUT_DIR, "day6_best.joblib")
    joblib.dump(best, out_path)
    print(f"Saved best model → {out_path}")

if __name__ == "__main__":
    main()
