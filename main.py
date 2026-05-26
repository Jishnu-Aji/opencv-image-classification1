# main.py
"""Simple Image Classification & Processing Demo

- Uses scikit-learn's digits dataset (default) – easy for beginners.
- Preprocesses images with OpenCV (grayscale, resize, normalize, contrast/brightness, Gaussian blur).
- Trains a k‑Nearest Neighbors classifier (k=3).
- Evaluates accuracy and displays a confusion matrix.
- Saves visualisations (original vs. enhanced, confusion matrix, sample predictions) to the `outputs/` folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Local imports
from utils import data_loader, preprocess, model, visualize


def ensure_output_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)


def main():
    # -------------------- Load Dataset --------------------
    X, y = data_loader.load_digits()

    # -------------------- Preprocess --------------------
    X_processed = np.array([preprocess.preprocess_image(img) for img in X])
    # Flatten each 32x32 image to a 1‑D vector for sklearn models
    X_flat = X_processed.reshape(len(X_processed), -1)

    # -------------------- Train / Test Split --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------- Model Training --------------------
    clf = model.train_knn(X_train, y_train, k=3)

    # -------------------- Evaluation --------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)

    # -------------------- Visualisations --------------------
    ensure_output_dir()
    # Original vs enhanced for a few samples
    for idx in range(5):
        original = X[idx]
        enhanced = preprocess.preprocess_image(original)
        visualize.save_original_vs_enhanced(original, enhanced, idx, "outputs")
    # Confusion matrix heat‑map
    visualize.save_confusion_matrix(cm, classes=np.arange(10), out_path="outputs/confusion_matrix.png")
    # Sample predictions with true labels
    visualize.save_predictions_grid(clf, X_test, y_test, y_pred, out_path="outputs/predictions.png", n=9)

    print("All visualisations saved to the 'outputs/' folder.")


if __name__ == "__main__":
    main()
