"""utils/preprocess.py
Utility functions for image preprocessing using OpenCV.
"""
import cv2
import numpy as np


def to_grayscale(img):
    """Convert image to grayscale if it is not already."""
    if len(img.shape) == 2:  # already grayscale
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img, size=(32, 32)):
    """Resize image to the given size (default 32x32)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_image(img):
    """Normalize pixel values to the range [0, 1]."""
    img = img.astype(np.float32)
    return img / 255.0


def enhance_contrast_brightness(img, alpha=1.2, beta=20):
    """Adjust contrast (alpha) and brightness (beta).
    alpha > 1 increases contrast, beta adds brightness.
    """
    # cv2.convertScaleAbs handles clipping and type conversion
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_gaussian_blur(img, ksize=(3, 3)):
    """Apply Gaussian blur with given kernel size."""
    return cv2.GaussianBlur(img, ksize, sigmaX=0)


def preprocess_image(img):
    """Full preprocessing pipeline applied to a single image.
    Steps:
        1. Grayscale conversion
        2. Resize to 32x32
        3. Contrast & brightness enhancement
        4. Gaussian blur
        5. Normalization to [0, 1]
    Returns the processed image as a float32 NumPy array.
    """
    gray = to_grayscale(img)
    resized = resize_image(gray)
    enhanced = enhance_contrast_brightness(resized)
    blurred = apply_gaussian_blur(enhanced)
    normalized = normalize_image(blurred)
    return normalized
