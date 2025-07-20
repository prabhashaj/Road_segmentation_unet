# config.py
import os

# Paths
data_dir = os.path.join('dataset', 'patches')
images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')
model_path = 'best_model.pth'

# Hyperparameters
batch_size = 1
epochs = 30
learning_rate = 1e-4
val_split = 0.2
img_size = 256
num_workers = 2

# Model
num_classes = 1  # Binary segmentation
in_channels = 3
