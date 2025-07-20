# visualize.py
import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for better interactive plotting
import matplotlib.pyplot as plt
from model import UNet
from utils import RoadDataset, get_image_mask_paths
import config

def visualize_predictions(num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=config.in_channels, out_channels=config.num_classes).to(device)
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {config.model_path} (checkpoint dictionary)")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {config.model_path} (plain state_dict)")
    else:
        print("No trained model found.")
        return
    model.eval()
    images, masks = get_image_mask_paths(config.images_dir, config.masks_dir)
    dataset = RoadDataset(images, masks, img_size=config.img_size)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for idx in indices:
        img, mask = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            pred = (pred > 0.5).astype(np.uint8)
        img_np = img.permute(1,2,0).numpy()
        mask_np = mask.squeeze().numpy()
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('Image')
        plt.imshow(img_np)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Ground Truth')
        plt.imshow(mask_np, cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Prediction')
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    visualize_predictions(num_samples=5)






