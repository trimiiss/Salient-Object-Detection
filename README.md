# 🎯 Salient Object Detection (SOD)

## 📌 Project Overview

This project implements a complete **end-to-end Salient Object Detection (SOD)** system.

Using a custom deep learning architecture, the system automatically **segments the most visually dominant objects** in an image.

The pipeline covers the full ML lifecycle:

* Data merging
* Preprocessing & augmentation
* Training with hybrid loss
* Evaluation & visualization

---

## 📊 Dataset & "Super-Set" Strategy

* **Source:** DUTS Dataset
* **Total Samples:** 15,572 image-mask pairs

### 🔹 Super-Set Approach

To maximize diversity and avoid bias from predefined splits:

* DUTS-TR and DUTS-TE were **merged into one dataset**
* A custom randomized split was applied:

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

---

## 🔄 Preprocessing & Augmentation

### Preprocessing

* Resize images to **224 × 224**
* Normalize pixel values to **[0,1]**

### Augmentations

* Random Horizontal Flip
* Random Vertical Flip
* Color Jitter
* Random Rotation (±15°)

---

## 🏗️ Model Architecture (SODNet)

The model follows a **symmetrical Encoder-Decoder design** optimized for spatial accuracy.

### 🔹 Encoder

* 5 blocks:

  * Conv → BatchNorm → ReLU → MaxPool
* Extracts deep semantic features
* Additional Conv layers for deeper abstraction

### 🔹 Decoder

* Uses **ConvTranspose2d** for upsampling
* Conv → BatchNorm → ReLU after each step

### 🔹 Skip Connections

* Feature concatenation between encoder & decoder
* Preserves fine details (edges, boundaries)

### 🔹 Regularization

* Dropout (0.2)
* Batch Normalization

---

## ⚙️ Training Setup

### 📉 Hybrid Loss Function

L = \mathrm{BCE} + 0.5(1 - \mathrm{IoU})

* Combines:

  * Pixel-wise accuracy (BCE)
  * Shape overlap (IoU)

### ⚙️ Configuration

* **Optimizer:** Adam
* **Learning Rate:** 1 × 10⁻³
* **Epochs:** 25
* **Early Stopping:** Based on validation loss

---

## 📈 Final Results & Metrics

Evaluated on an unseen test set (**2,336 images**):

| Metric    | Score  | Analysis               |
| --------- | ------ | ---------------------- |
| IoU       | 0.7079 | High spatial overlap   |
| Precision | 0.7853 | Low false positives    |
| Recall    | 0.8699 | Strong object coverage |
| F1-Score  | 0.8044 | Balanced performance   |
| MAE       | 0.0958 | Low pixel error        |

---

## 🖼 Visual Demonstration

Each prediction includes:

1. Input Image
2. Ground Truth
3. Predicted Heatmap
4. Overlay Result

---

## 🏗 Repository Structure
├── checkpoints/            # Trained model weights (best.pt)  
├── visualizations/         # Output results and grids  
├── data_loader.py          # Data pipeline (resize, normalize, augmentations)  
├── sod_model.py            # SODNet architecture (Encoder-Decoder CNN)  
├── train.py                # Training loop (BCE + IoU optimization)  
├── evaluate.py             # Evaluation metrics (IoU, F1, MAE)  
├── generate_images.py      # Inference on single images (demo)  
├── training_log.json       # Training history across epochs  
├── requirements.txt        # Dependencies list  

## ⚙️ Installation

```bash id="install-steps"
# Create virtual environment
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train

```bash id="train-cmd"
python train.py
```

### Evaluate

```bash id="eval-cmd"
python evaluate.py
```

### Generate Visual Results

```bash id="vis-cmd"
python generate_grid.py
```

---

## 🧠 Key Insights

* High **Recall (0.8699)** → captures full object regions
* Strong **IoU (0.7079)** → accurate segmentation
* Balanced **F1-score** → reliable performance overall

---

## 🎓 Conclusion

The model demonstrates strong robustness in detecting salient objects even in complex scenes.

### 📌 Applications

* Background removal
* Autonomous driving
* Medical image segmentation

---

## 🚀 Future Improvements

* Attention U-Net
* Transformer-based encoders
* Real-time inference optimization

---

## ⭐ Author

**Trimi**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐
