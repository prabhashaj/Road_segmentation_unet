# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import UNet
from utils import get_loaders, dice_coeff
import config


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=config.in_channels, out_channels=config.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_dice = 0.0

    # Load existing weights if available
    if os.path.exists(config.model_path):
        print(f"Loading weights from {config.model_path}")
        checkpoint = torch.load(config.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_dice = checkpoint.get('best_dice', 0.0)
        else:
            # Assume it's a plain state_dict
            model.load_state_dict(checkpoint)
            print("Loaded plain state_dict (no optimizer/epoch info). Training will start from scratch.")

    train_loader, val_loader = get_loaders(
        config.images_dir, config.masks_dir, config.batch_size, config.val_split, config.img_size, config.num_workers
    )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coeff(outputs, masks) * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'best_dice': best_dice
            }, config.model_path)
            print(f"Best model saved with Dice {best_dice:.4f}")

if __name__ == '__main__':
    train()
