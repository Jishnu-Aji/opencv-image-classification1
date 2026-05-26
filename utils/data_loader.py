from sklearn.datasets import load_digits as sklearn_load_digits

def load_digits():
    """Load scikit‑learn's handwritten digits dataset.

    Returns:
        X (np.ndarray): Array of shape (n_samples, 8, 8) with pixel values 0‑16.
        y (np.ndarray): Integer labels 0‑9.
    """
    digits = sklearn_load_digits()
    X = digits.images  # shape (n_samples, 8, 8)
    y = digits.target
    return X, y

# Placeholder for CIFAR‑10 loader – not used in the default flow.
# You can extend this function to download CIFAR‑10 via keras or torchvision.

def load_cifar10_dataset():
    raise NotImplementedError("CIFAR‑10 loading not implemented in this starter project.")
