"""utils/model.py
Utility functions for model creation and training.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_knn(X_train, y_train, k=3):
    """Train a k‑Nearest Neighbors classifier.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        k (int): Number of neighbors (default 3).
    Returns:
        sklearn.neighbors.KNeighborsClassifier: Trained model.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def train_svm(X_train, y_train, C=1.0, kernel='rbf'):
    """Train a Support Vector Machine classifier.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        C (float): Regularization parameter.
        kernel (str): Kernel type (default 'rbf').
    Returns:
        sklearn.svm.SVC: Trained model.
    """
    svm = SVC(C=C, kernel=kernel, probability=True)
    svm.fit(X_train, y_train)
    return svm
