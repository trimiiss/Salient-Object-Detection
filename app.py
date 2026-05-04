"""
app.py
======
Gradio demo for Salient Object Detection.
Run with:  python app.py
Or in Colab:  !python app.py
"""

import time
import numpy as np
import torch
import cv2
import gradio as gr
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from sod_model import SODNet, SODNetPlus

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CHECKPOINT_BASELINE  = "checkpoints/best_baseline.pt"
CHECKPOINT_IMPROVED  = "checkpoints/best_improved.pt"
IMAGE_SIZE = 224
THRESHOLD  = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
def load_model(model_type="baseline"):
    ckpt_path = CHECKPOINT_BASELINE if model_type == "baseline" else CHECKPOINT_IMPROVED
    model = SODNet() if model_type == "baseline" else SODNetPlus()
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Demo] Loaded {ckpt_path}")
    else:
        print(f"[Demo] WARNING: checkpoint not found ({ckpt_path}). Using untrained model.")
    model.to(DEVICE).eval()
    return model

models = {
    "baseline": load_model("baseline"),
    "improved": load_model("improved"),
}


# ─────────────────────────────────────────────
# Inference function
# ─────────────────────────────────────────────
def predict(input_image, model_choice: str, threshold: float):
    if input_image is None:
        return None, None, None, "No image provided."

    # Pre-process
    img_rgb  = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) if input_image.shape[-1] == 3 else input_image
    orig_h, orig_w = img_rgb.shape[:2]

    aug     = transform(image=img_rgb)
    tensor  = aug["image"].unsqueeze(0).to(DEVICE)     # (1,3,H,W)

    # Inference with timing
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = models[model_choice](tensor)             # (1,1,H,W)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Post-process
    pred_np = pred.squeeze().cpu().numpy()              # (H,W)  values 0-1
    mask_bin = (pred_np > threshold).astype(np.uint8) * 255

    # Resize prediction back to original image size
    pred_resized    = cv2.resize(pred_np,  (orig_w, orig_h))
    mask_resized    = cv2.resize(mask_bin, (orig_w, orig_h))
    mask_bin_full   = (mask_resized > 127).astype(np.uint8)

    # Heatmap visualization
    heatmap = (pred_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay: tint salient region
    overlay = img_rgb.copy().astype(np.float32)
    overlay[mask_bin_full == 1] = overlay[mask_bin_full == 1] * 0.5 + np.array([255, 60, 60]) * 0.5
    overlay = overlay.clip(0, 255).astype(np.uint8)

    info = (
        f"Model      : {model_choice.upper()}\n"
        f"Threshold  : {threshold:.2f}\n"
        f"Inference  : {elapsed_ms:.1f} ms\n"
        f"Device     : {DEVICE}\n"
        f"Image size : {orig_w}×{orig_h}"
    )

    return heatmap_color, overlay, mask_resized, info


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
with gr.Blocks(title="SOD Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Salient Object Detection Demo
    Upload any image — the model highlights the most visually dominant region.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp_img   = gr.Image(label="Input Image", type="numpy")
            model_sel = gr.Radio(["baseline", "improved"], value="baseline",
                                 label="Model")
            thresh    = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                  label="Threshold")
            run_btn   = gr.Button("Run Inference", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                out_heat    = gr.Image(label="Saliency Heatmap")
                out_overlay = gr.Image(label="Overlay (prediction on image)")
            with gr.Row():
                out_mask    = gr.Image(label="Binary Mask")
                out_info    = gr.Textbox(label="Inference Info", lines=6)

    run_btn.click(
        fn=predict,
        inputs=[inp_img, model_sel, thresh],
        outputs=[out_heat, out_overlay, out_mask, out_info],
    )

    gr.Examples(
        examples=[[f"examples/{f}"] for f in Path("examples").glob("*.jpg")]
        if Path("examples").exists() else [],
        inputs=[inp_img],
    )


if __name__ == "__main__":
    demo.launch(share=True)   # share=True works in Colab
