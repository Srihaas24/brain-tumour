# 🧠 Brain Tumor MRI Classification & Training Guide

This repository contains the end-to-end pipeline for training deep learning models on the **BRISC 2025 (Brain tumor Image Segmentation & Classification)** T1-weighted MRI dataset, and deploying them to an interactive medical diagnosis web interface.

---

## 📁 Repository & Dataset Structure

The workspace has been organized into a clean, modular structure:

```
c:\Srihaas\ALTHON/
├── dataset/
│   ├── classification/
│   │   ├── train/
│   │   │   ├── glioma/          # ~1,250 axial/coronal/sagittal T1 slices
│   │   │   ├── meningioma/      # ~1,250 slices
│   │   │   ├── no_tumor/        # ~1,250 normal healthy brain slices
│   │   │   └── pituitary/       # ~1,250 pituitary adenoma slices
│   │   └── test/
│   │       ├── glioma/          # 250 slices
│   │       ├── meningioma/      # 250 slices
│   │       ├── no_tumor/        # 250 slices
│   │       └── pituitary/       # 250 slices
│   ├── segmentation/
│   │   ├── train/               # Paired images & expert radiologist masks
│   │   └── test/
│   ├── manifest.csv             # Full metadata manifest with patient & plane tags
│   └── DATASET_README.md
├── training/
│   ├── dataset.py               # Data loaders, augmentation & preprocessing
│   ├── model.py                 # Neural network architecture & Grad-CAM hooks
│   ├── train.py                 # Training script with checkpointing & curves
│   ├── evaluate.py              # Confusion matrix & metrics report
│   └── README.md
├── models/
│   ├── best_model.pt            # Checkpoint with best validation accuracy
│   ├── model_metadata.json      # Model hyperparameters and performance summary
│   ├── training_curves.png      # Loss and accuracy graphs
│   └── confusion_matrix.png     # Validation / test confusion matrix
├── app/
│   ├── server.py                # Flask application & prediction API
│   ├── static/
│   │   ├── css/style.css        # Premium UI design styling
│   │   └── js/main.js           # Client-side reactivity & image processing
│   └── templates/
│       └── index.html           # Diagnostic dashboard
└── requirements.txt
```

---

## 🏷 Classification Classes

The model classifies scans into 4 mutually exclusive categories:
1. **Glioma** (`gl`): Infiltrative intrinsic brain tumors arising from glial cells. High vs low grade.
2. **Meningioma** (`me`): Typically extra-axial, slow-growing tumors arising from the protective meninges.
3. **Pituitary Tumor** (`pi`): Tumors arising in the sella turcica, commonly causing endocrine or visual disturbance.
4. **No Tumor** (`no_tumor`): Healthy brain scans without space-occupying mass or abnormal enhancement.

---

## 🚀 How to Train the Model

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
To start training using the recommended lightweight, high-performance **MobileNetV3** backbone:

```bash
python training/train.py --epochs 10 --batch-size 32 --lr 0.0005 --backbone mobilenet_v3
```

#### Available CLI Arguments:
| Argument | Default | Description |
|---|---|---|
| `--epochs` | `5` | Number of training iterations over the dataset |
| `--batch-size` | `32` | Number of MRI slices per optimization batch |
| `--lr` | `0.0005` | Initial learning rate (uses Cosine Annealing) |
| `--backbone` | `mobilenet_v3` | Feature extractor: `mobilenet_v3`, `resnet18`, or `custom` |
| `--subset` | `1.0` | Fraction of data to use (e.g. `0.2` for quick 20% smoke tests) |
| `--device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `--model-dir` | `models` | Output folder for `.pt` weights and curve plots |

---

## 📊 How to Evaluate the Model

Once training is complete, generate a complete confusion matrix and precision/recall report on the 1,000-image test set:

```bash
python training/evaluate.py --model-path models/best_model.pt
```

Outputs generated in `models/`:
- `confusion_matrix.png`: Heatmap comparing actual vs predicted diagnoses.
- `test_evaluation.json`: Exact accuracy, sensitivity, and specificity figures.

---

## 🌐 Running the Web Application

Launch the local diagnostic interface:

```bash
python app/server.py
```
Then open your web browser at:
`http://localhost:5000`
