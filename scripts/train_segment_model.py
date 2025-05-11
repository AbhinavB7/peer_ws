import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Dataset class
class GroundSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        return img, mask

# Transforms
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# Function to collect image-mask pairs
def get_image_mask_pairs(image_dir, mask_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    image_paths = []
    mask_paths = []
    for img_file in sorted(image_files):
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}_mask.png"
        if os.path.exists(os.path.join(mask_dir, mask_file)):
            image_paths.append(os.path.join(image_dir, img_file))
            mask_paths.append(os.path.join(mask_dir, mask_file))
        else:
            print(f"Warning: Mask not found for {img_file}")
    return image_paths, mask_paths

# Define your dataset folders
train_image_dir = "../datasets/Segmentation/train/images"
train_mask_dir = "../datasets/Segmentation/train/masks"
val_image_dir = "../datasets/Segmentation/val/images"
val_mask_dir = "../datasets/Segmentation/val/masks"

# Load image-mask pairs
train_imgs, train_masks = get_image_mask_pairs(train_image_dir, train_mask_dir)
val_imgs, val_masks = get_image_mask_pairs(val_image_dir, val_mask_dir)

# Dataset & Loaders
train_dataset = GroundSegDataset(train_imgs, train_masks, transform=train_transform)
val_dataset = GroundSegDataset(val_imgs, val_masks, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model
# model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=1, activation=None)

model = model.cuda()

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/segmentation")

# Training loop
for epoch in range(20):
    print(f"\nEpoch {epoch+1}/20")

    # Training
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc="Training"):
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validating"):
            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs)
            val_loss += loss_fn(preds, masks).item()

        # Log one sample every 5 epochs
        if (epoch + 1) % 2 == 0:
            input_img = imgs[0].cpu()
            gt_mask = masks[0].cpu()
            pred_mask = torch.sigmoid(preds[0]).cpu().clamp(0, 1)
            writer.add_image("Image", input_img, epoch + 1)
            writer.add_image("GT Mask", gt_mask, epoch + 1)
            writer.add_image("Predicted Mask", pred_mask, epoch + 1)

    # Compute average losses
    train_epoch_loss = train_loss / len(train_loader)
    val_epoch_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    # Log scalars to TensorBoard
    writer.add_scalar("Loss/Train", train_epoch_loss, epoch + 1)
    writer.add_scalar("Loss/Val", val_epoch_loss, epoch + 1)

# Save final model
torch.save(model.state_dict(), "../models/segmentation/mobilenet.pth")
print("Training complete. Model saved.")
