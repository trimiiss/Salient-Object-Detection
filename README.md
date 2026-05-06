# 🎯 Salient Object Detection (SOD) using Encoder-Decoder CNN

## 📌 Project Overview

This project implements a Deep Learning system designed to identify and segment the most visually **salient (important) objects** within an image.

Using a custom **U-Net inspired architecture**, the model converts raw RGB images into precise binary segmentation masks.

---

## 📊 Key Performance Metrics

| Metric   | Score  |
| -------- | ------ |
| IoU      | 0.7079 |
| F1-Score | 0.8044 |
| MAE      | 0.0958 |
| Recall   | 0.8699 |

---

## 🏗️ System Architecture

The model (**SODNet**) follows an **Encoder-Decoder structure**:

### 🔹 Encoder

* 5 convolutional blocks
* Each block includes:

  * Convolution
  * Batch Normalization
  * ReLU activation
* Reduces spatial resolution while extracting features

### 🔹 Bottleneck

* 1024-channel latent representation
* Captures high-level semantic information

### 🔹 Decoder

* Uses `ConvTranspose2d` for upsampling
* Restores spatial resolution back to **224 × 224**

### 🔹 Skip Connections

* Implemented using `torch.cat`
* Helps preserve fine details (edges, object boundaries)

---

## 📊 Qualitative Results

Each output visualization includes:

1. Input Image
2. Ground Truth
3. Predicted Heatmap
4. Saliency Overlay

---

## 📂 Repository Structure

```
├── sod_model.py        # Model architecture (SODNet)
├── data_loader.py      # Dataset handling + augmentations
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation metrics
├── generate_grid.py    # Visualization generator
├── training_log.json   # Training history (25 epochs)
├── checkpoints/
│   └── best.pt         # Trained model weights
└── requirements.txt    # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sod-project.git
cd sod-project
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Train the Model

```bash
python train.py
```

### 🔹 Evaluate the Model

```bash
python evaluate.py
```

### 🔹 Generate Visual Results

```bash
python generate_grid.py
```

---

## 📦 Dataset

* Uses **DUTS Dataset**
* Includes:

  * Training images
  * Ground truth masks
* Data augmentations:

  * Random Flip
  * Rotation
  * Color Jitter

---

## 🎓 Conclusion

The model demonstrates strong performance in detecting salient objects even in complex scenes.

* High **Recall (0.8699)** → captures full object regions effectively
* Suitable for:

  * Background removal
  * Autonomous driving
  * Medical image segmentation

---

## 📌 Future Improvements

* Attention mechanisms (Attention U-Net)
* Transformer-based encoders
* Real-time inference optimization

---

## 🧠 Author

**Trimi**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
