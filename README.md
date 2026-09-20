# Dogs vs Cats Image Classification (PyTorch)

A fully-connected neural network (MLP) built in PyTorch to classify images of dogs and cats.

## Overview

This project trains an MLP on the classic Dogs vs Cats image dataset, resizing images to 64×64 and feeding flattened pixel values through a fully-connected network — a baseline before moving to convolutional architectures.

## Dataset

- **Source:** `dogs-vs-cats-classification` (uploaded as a zip archive and extracted in Colab)
- **Classes:** `cats`, `dogs`
- **Split:** pre-organized into `train/`, `validation/`, `test/` folders (via `ImageFolder`)
  - Train: 19,943 images
  - Validation: 2,492 images
  - Test: 2,495 images

> Note: the notebook uploads and extracts the dataset zip via Google Colab's file upload widget. Update this step (e.g. point directly to a local folder) if running outside Colab.

## Approach

1. **Preprocessing:**
   - Resize images to 64×64
   - Training set: random horizontal flip (data augmentation) + normalization
   - Validation/test sets: normalization only (no augmentation)
   - Loaded via `torchvision.datasets.ImageFolder` + `DataLoader` (`batch_size=32`)
2. **Model architecture** (`MLPModel`, fully connected — images flattened to vectors):
   - Flatten(3×64×64) → Linear(512) → ReLU → Dropout(0.5) → Linear(128) → ReLU → Dropout(0.3) → Linear(2)
3. **Training:**
   - Loss: `CrossEntropyLoss`
   - Optimizer: Adam, `learning_rate=0.001`
   - Epochs: 10, with per-epoch train/validation loss and accuracy tracked
4. **Evaluation:** Accuracy and full classification report on the test set
5. **Inference:** Loads a single new image, applies the same preprocessing, and predicts its class

## Results

| Metric | Value |
|---|---|
| Final Train Accuracy (epoch 10) | 61.78% |
| Final Validation Accuracy (epoch 10) | ~62% |
| **Test Accuracy** | **63.17%** |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| cats | 0.68 | 0.50 | 0.57 |
| dogs | 0.60 | 0.77 | 0.68 |

An MLP on flattened pixels is a weak architecture for images — it discards all spatial structure, which is why accuracy plateaus around 60–63% despite a large training set. The model is noticeably better at recalling dogs than cats.

## Requirements

```
torch
torchvision
pillow
matplotlib
scikit-learn
```

Install with:
```bash
pip install torch torchvision pillow matplotlib scikit-learn
```

## Usage

1. Obtain the Dogs vs Cats dataset organized into `train/`, `validation/`, `test/` folders, each containing `cats/` and `dogs/` subfolders (e.g. from [Kaggle's Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats)).
2. Update `train_dir`, `val_dir`, `test_dir` to point to your local paths (skip the Colab upload/zip-extraction cells if running locally).
3. Run the notebook top to bottom.
4. To classify a new image:

```python
from PIL import Image
import torch

image = Image.open("path/to/image.jpg").convert("RGB")
image = val_test_transform(image).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Prediction:", test_dataset.classes[predicted.item()])
```

## Project Structure

```
.
├── Untitled22.ipynb                      # Data loading, model, training, evaluation, inference
├── dogs-vs-cats-classification/          # Dataset: train/validation/test folders (add your own)
└── README.md
```

*(Consider renaming the notebook to something like `Dogs_vs_Cats_MLP.ipynb` for clarity in the repo.)*

## Future Improvements

- Replace the MLP with a **CNN** — convolutional layers preserve spatial structure and should substantially improve accuracy on this task
- Try transfer learning with a pretrained model (ResNet, MobileNet) for a strong accuracy boost with less training
- Increase image resolution beyond 64×64 (with a CNN, higher resolution tends to help)
- Add more data augmentation (rotation, color jitter, random crop)
- Track and plot train/validation loss and accuracy curves across epochs
