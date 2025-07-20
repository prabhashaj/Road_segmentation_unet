# Patch extraction script for Massachusetts Roads Dataset
import os
from PIL import Image
import numpy as np

PATCH_SIZE = 256
# Source folders
SPLITS = ['train', 'val', 'test']
SRC_ROOT = os.path.join('dataset', 'tiff')
DST_IMG = os.path.join('dataset', 'patches', 'images')
DST_MASK = os.path.join('dataset', 'patches', 'masks')
os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_MASK, exist_ok=True)

def extract_and_save_patches(split):
    img_dir = os.path.join(SRC_ROOT, split)
    mask_dir = os.path.join(SRC_ROOT, f'{split}_labels')
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif') or f.endswith('.tiff')])
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        # Try to find mask with any extension
        mask_base = os.path.splitext(fname)[0]
        mask_path = None
        for ext in ['.tif', '.tiff', '.png']:
            candidate = os.path.join(mask_dir, mask_base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f'Mask not found for {fname}, skipping.')
            continue
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        h, w = img.shape[:2]
        for i in range(0, h, PATCH_SIZE):
            for j in range(0, w, PATCH_SIZE):
                img_patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                if img_patch.shape[0] == PATCH_SIZE and img_patch.shape[1] == PATCH_SIZE:
                    if np.any(mask_patch):  # skip empty patches
                        patch_name = f'{split}_{mask_base}_{i}_{j}.png'
                        Image.fromarray(img_patch).save(os.path.join(DST_IMG, patch_name))
                        Image.fromarray((mask_patch > 0).astype(np.uint8)*255).save(os.path.join(DST_MASK, patch_name))
                        
if __name__ == '__main__':
    for split in SPLITS:
        print(f'Processing {split}...')
        extract_and_save_patches(split)
    print('Patch extraction complete!')
