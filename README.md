# Simple Image Classification & Image Processing with OpenCV

![Project Hero](file:///C:/Users/user/.gemini/antigravity-ide/brain/7c11a422-e92e-47e1-ba93-aa40e7ec3fcc/image_classification_hero_1779805744206.png)

## Project Overview
A beginner‑friendly Python project that demonstrates how to:
- Load a classic image dataset (scikit‑learn's handwritten digits).
- Preprocess images with OpenCV (grayscale, resize, contrast/brightness enhancement, Gaussian blur, normalization).
- Train a machine‑learning model (k‑Nearest Neighbors or SVM).
- Evaluate performance using accuracy and a confusion matrix.
- Visualise original vs. enhanced images and sample predictions.

The code is intentionally simple, heavily commented, and organized into reusable utility modules.

## Technologies Used
- **Python** (>=3.8)
- **OpenCV** – image processing (`opencv-python`)
- **NumPy** – numerical operations
- **Matplotlib** – plotting and saving figures
- **scikit‑learn** – dataset, model implementations, evaluation metrics

## Dataset Information
- **Digits dataset** – 1797 8×8 grayscale images of handwritten digits (0‑9) available via `sklearn.datasets.load_digits`.
- (Optional) **CIFAR‑10** – placeholder for future extension.

## Setup & Installation
1. **Clone / copy** this repository to your local machine.
2. Open a terminal in the project root (`cv-1`).
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
```bash
python main.py
```
The script will:
1. Load the digits dataset.
2. Apply the full OpenCV preprocessing pipeline.
3. Split data (80 % train / 20 % test).
4. Train a **k‑Nearest Neighbors (k=3)** classifier.
5. Print accuracy and the confusion matrix.
6. Save visualisations to the `outputs/` folder:
   - `original_vs_enhanced_*.png`
   - `confusion_matrix.png`
   - `predictions.png`

## Results
Typical output (may vary slightly due to random split):
```
Accuracy: 0.9667
Confusion Matrix:
 [[...]]
All visualisations saved to the 'outputs/' folder.
```
*The confusion matrix heat‑map and sample prediction grid are saved as PNG images in `outputs/`.*

## Folder Structure
```
cv-1/
├─ main.py                # entry point
├─ requirements.txt       # package list
├─ README.md              # you are reading it!
├─ outputs/               # generated figures
└─ utils/
   ├─ __init__.py         # makes utils a package
   ├─ data_loader.py      # dataset loading helpers
   ├─ preprocess.py       # OpenCV preprocessing pipeline
   ├─ model.py            # k‑NN / SVM training helpers
   └─ visualize.py        # image saving & plotting utilities
```

## License
MIT License – feel free to modify and share.
