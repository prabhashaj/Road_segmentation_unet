# utils.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def get_image_mask_paths(images_dir, masks_dir):
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
    masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')])
    return images, masks

class RoadDataset(Dataset):
    def __init__(self, images, masks, img_size=256):
        self.images = images
        self.masks = masks
        self.img_size = img_size
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(self.masks[idx]).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
        img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        return img, mask

def get_loaders(images_dir, masks_dir, batch_size, val_split, img_size, num_workers):
    images, masks = get_image_mask_paths(images_dir, masks_dir)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    train_imgs = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]
    val_imgs = [images[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]
    train_ds = RoadDataset(train_imgs, train_masks, img_size)
    val_ds = RoadDataset(val_imgs, val_masks, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def dice_coeff(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()
