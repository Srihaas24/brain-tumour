# 🧠 NeuroScan AI — Brain Tumor Multi-Task Detection & Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1keF7ZA-2gEtl4dHBHhGVy_Ay__eOXrhE?usp=sharing)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

An end-to-end deep learning diagnostic system for **simultaneous brain tumor classification and pixel-level lesion segmentation** using the **BRISC 2025** benchmark dataset (T1-weighted MRI).

---

## 🌟 Key Features

1. **Multi-Task Neural Network:**
   - Single shared encoder (**ResNet-34** / **nnU-Net** style).
   - **Classification Head:** Predicts tumor pathology (**Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**).
   - **Segmentation Head:** Outputs high-resolution binary mask delineating exact lesion margins.
2. **Clinical Visualizer:**
   - Original T1 MRI scan.
   - Doctor/Radiologist ground truth mask.
   - Predicted binary mask.
   - Translucent color overlay comparison with **Dice Score %**.
3. **Interactive Web Dashboard:**
   - Drag-and-drop MRI upload.
   - 1-click sample scan test gallery.
   - Grad-CAM and mask opacity cross-fader sliders.
   - Printable diagnostic medical report.

---

## 🚀 Run in Google Colab (Free GPU)

You can train and evaluate the full multi-task pipeline in Google Colab with 1 click:

👉 **[Open Notebook in Google Colab](https://colab.research.google.com/drive/1keF7ZA-2gEtl4dHBHhGVy_Ay__eOXrhE?usp=sharing)**

*(The notebook downloads the BRISC 2025 dataset directly via Figshare CDN in ~3 seconds—no Google Drive upload needed).*

---

## 💻 Local Setup & Web App

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app/server.py
```
Open **`http://localhost:5000`** in your browser.

---

## 📁 Repository Structure

```
.
├── brain_tumor_multitask_colab.ipynb  # Interactive Google Colab notebook
├── colab_train.py                     # Standalone Python script for training
├── requirements.txt                   # Dependencies
├── .gitignore                         # Excludes large raw datasets and checkpoints
├── app/
│   ├── server.py                      # Flask REST API backend
│   ├── templates/index.html           # Diagnostic UI dashboard
│   └── static/                        # CSS, JS, and sample MRI library
├── training/
│   ├── dataset.py                     # Multi-task dataset loader & augmentations
│   ├── model.py                       # Model architectures & Grad-CAM hooks
│   ├── train.py                       # Modular local training pipeline
│   └── evaluate.py                    # Confusion matrix & metrics
└── models/                            # Trained weights & metadata
```

---

## 📊 Dataset Reference

Based on the **BRISC 2025** (Brain Tumor Image Segmentation & Classification) dataset:
- **Total Slices:** 6,000 contrast-enhanced T1 MRI scans (5,000 train / 1,000 test).
- **Classes:** Glioma, Meningioma, Pituitary, Normal.
- **Annotations:** Pixel-level masks reviewed and corrected by certified radiologists.
- **DOI:** [10.6084/m9.figshare.30533120](https://doi.org/10.6084/m9.figshare.30533120)
