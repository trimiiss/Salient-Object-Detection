# Salient Object Detection — End-to-End ML Project
### Cohort V | Project #3

---

## Project Structure

```
sod_project/
├── data_loader.py          ← Dataset loading, preprocessing, augmentation
├── sod_model.py            ← CNN models (SODNet baseline + SODNetPlus improved)
├── train.py                ← Training loop with checkpoint + resume support
├── evaluate.py             ← Metrics (IoU, Precision, Recall, F1) + visualizations
├── app.py                  ← Gradio demo app
├── SOD_Project.ipynb       ← Master Colab notebook (run everything from here)
├── requirements.txt
├── dataset/                ← Put your dataset here (created at runtime)
│   ├── images/
│   └── masks/
├── checkpoints/            ← Saved model weights (created at runtime)
└── outputs/                ← Evaluation visualizations (created at runtime)
```

---

## Quick Start (Google Colab)

1. Upload `sod_project.zip` to Colab
2. Open `SOD_Project.ipynb`
3. Run cells top to bottom — each section is self-contained

---

## Models

| Model | Architecture | Features |
|---|---|---|
| `SODNet` (baseline) | Encoder-Decoder CNN | 4 conv blocks, no skip connections |
| `SODNetPlus` (improved) | U-Net style | BatchNorm + Dropout + skip connections |

---

## Loss Function

```
Loss = BCE(pred, target) + 0.5 × (1 − IoU(pred, target))
```

---

## Dataset Format

```
dataset/
  images/   ←  .jpg or .png RGB images
  masks/    ←  .png grayscale binary masks (same filename as images)
```

Supported datasets: **DUTS**, **ECSSD**, **MSRA10K**

---

## Evaluation Metrics

- **IoU** — Intersection over Union  
- **Precision** — TP / (TP + FP)  
- **Recall** — TP / (TP + FN)  
- **F1-Score** — Harmonic mean of Precision & Recall  
- **MAE** — Mean Absolute Error (pixel-level)

---

## Checkpoint Resume (Bonus Feature)

Training automatically saves `checkpoints/latest.pt` after every epoch.  
If Colab disconnects, simply re-run the training cell — it will resume from the last epoch automatically.

---

## Author

> Replace with your name and cohort details before submission.
