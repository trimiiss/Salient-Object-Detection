import torch
import matplotlib.pyplot as plt
from PIL import Image
from sod_model import SODNet # This imports your architecture
from torchvision import transforms

def save_for_report(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SODNet().to(device)
    model.load_state_dict(torch.load('checkpoints/best.pt', map_location=device))
    model.eval()

    # Pre-process
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor)).squeeze().cpu().numpy()

    # Plot & Save
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original Image")
    plt.subplot(1, 3, 2); plt.imshow(pred, cmap='gray'); plt.title("Saliency Mask")
    plt.subplot(1, 3, 3); plt.imshow(img.resize((224, 224))); plt.imshow(pred, cmap='jet', alpha=0.4); plt.title("Final Overlay")
    
    plt.savefig('report_visual_1.png') # This goes straight into your Word Doc!
    plt.show()

# Run it
save_for_report('test_image.jpg')