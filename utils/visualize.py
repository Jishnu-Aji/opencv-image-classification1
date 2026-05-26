"""utils/visualize.py
Utility functions for visualising preprocessing results and model performance.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools


def save_original_vs_enhanced(original, enhanced, idx, out_dir):
    """Save a side‑by‑side image showing the original and the enhanced version.
    Args:
        original (np.ndarray): Original image (8x8 or 32x32).
        enhanced (np.ndarray): Processed image (float32 in [0,1]).
        idx (int): Index used for naming the file.
        out_dir (str): Directory where the image will be saved.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Resize both images to 32x32 for side‑by‑side comparison
    disp_original = cv2.resize(original, (32, 32))
    disp_enhanced = cv2.resize(enhanced, (32, 32))
    combined = np.hstack((disp_original, disp_enhanced))
    # Plot and save using matplotlib for better handling of colormaps
    plt.figure(figsize=(6, 3))
    plt.imshow(combined, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_{idx}.png", dpi=150)
    plt.close()


def save_confusion_matrix(cm, classes, out_path):
    """Save a heat‑map of the confusion matrix.
    Args:
        cm (np.ndarray): Confusion matrix.
        classes (list or array): Class labels.
        out_path (str): Output file path.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_predictions_grid(model, X_test, y_true, y_pred, out_path, n=9):
    """Save a grid of sample predictions.
    Shows the image, predicted label and true label.
    Args:
        model: Trained classifier (unused but kept for future extensions).
        X_test (np.ndarray): Test feature matrix (flattened).
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        out_path (str): Output PNG path.
        n (int): Number of samples to display (grid will be sqrt(n) x sqrt(n)).
    """
    plt.figure(figsize=(8, 8))
    indices = np.random.choice(len(X_test), n, replace=False)
    grid_size = int(np.ceil(np.sqrt(n)))
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(32, 32)
        label_true = y_true[idx]
        label_pred = y_pred[idx]
        ax = plt.subplot(grid_size, grid_size, i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T:{label_true} P:{label_pred}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
