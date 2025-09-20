# day9_cnn_train.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers # pyright: ignore[reportMissingImports]

DATA_PATH = "features/cnn_dataset.npz"
MODEL_DIR = "features/models"
PLOT_DIR  = "features/plots"
AN_DIR    = "features/analysis"
RNG = 42
BATCH = 16
EPOCHS = 25

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(AN_DIR, exist_ok=True)

def build_model(input_shape):
    inp = layers.Input(shape=input_shape)            # (64,128,1)
    x = layers.Conv2D(16, 3, padding="same")(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same")(x);     x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same")(x);     x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    d = np.load(DATA_PATH, allow_pickle=True)
    X, y, files = d["X"], d["y"].astype(int), d["files"]
    print("Dataset:", X.shape, y.shape)

    strat = y if np.min(np.bincount(y)) >= 1 else None
    Xtr, Xval, ytr, yval, ftr, fval = train_test_split(
        X, y, files, test_size=0.2, random_state=RNG, stratify=strat
    )

    # Save validation file list for later comparison with Day 6 model
    with open(os.path.join(AN_DIR, "day9_val_files.txt"), "w") as f:
        for fp in fval: f.write(fp + "\n")

    # Optional: class weights for imbalance
    counts = np.bincount(y)
    cw = {0: len(y)/(2*counts[0]), 1: len(y)/(2*counts[1])}

    model = build_model(X.shape[1:])
    ckpt = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "day9_cnn_best.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    )
    es = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=cw,
        callbacks=[ckpt, es],
        verbose=1
    )

    # Curves
    plt.figure(figsize=(7,4))
    plt.plot(hist.history["accuracy"], label="train acc")
    plt.plot(hist.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("CNN Accuracy")
    plt.legend(); plt.tight_layout()
    acc_path = os.path.join(PLOT_DIR, "day9_cnn_acc.png")
    plt.savefig(acc_path, dpi=200); plt.show()
    print("Saved:", acc_path)

    plt.figure(figsize=(7,4))
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("CNN Loss")
    plt.legend(); plt.tight_layout()
    loss_path = os.path.join(PLOT_DIR, "day9_cnn_loss.png")
    plt.savefig(loss_path, dpi=200); plt.show()
    print("Saved:", loss_path)

    # Validation performance + confusion matrix
    yhat = (model.predict(Xval) > 0.5).astype(int).ravel()
    acc  = accuracy_score(yval, yhat)
    print(f"Validation Accuracy (CNN): {acc:.3f}")
    print(classification_report(yval, yhat, target_names=["Noise (0)","Speech (1)"], digits=3))

    cm = confusion_matrix(yval, yhat, labels=[0,1])
    ConfusionMatrixDisplay(cm, display_labels=["Noise","Speech"]).plot(cmap="Blues")
    plt.title("Confusion Matrix – Day 9 CNN"); plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, "day9_cnn_confusion.png")
    plt.savefig(cm_path, dpi=200); plt.show()
    print("Saved:", cm_path)

if __name__ == "__main__":
    main()
