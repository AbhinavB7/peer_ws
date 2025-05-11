import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from tqdm import tqdm

# Minimal Dataset Class
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

# Transformation
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

# Match image-mask pairs from test folder
def get_image_mask_pairs(image_dir, mask_dir):
    image_paths, mask_paths = [], []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".jpg"):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, base + "_mask.png")
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print(f"Skipping {fname}: mask not found.")
    return image_paths, mask_paths

# Define test paths
test_image_dir = "../datasets/Segmentation/test/images"
test_mask_dir = "../datasets/Segmentation/test/masks"
test_imgs, test_masks = get_image_mask_pairs(test_image_dir, test_mask_dir)

# Dataset & Loader
test_dataset = GroundSegDataset(test_imgs, test_masks, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load trained model
# model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
model = smp.Unet(encoder_name="mobilenet_v2", in_channels=3, classes=1, activation=None)

model.load_state_dict(torch.load("../models/segmentation/mobilenet.pth"))
model = model.cuda()
model.eval()

# Metrics
def compute_metrics(preds, masks):
    preds_bin = (preds > 0.5).float()
    iou = jaccard_score(masks.view(-1).cpu().numpy(), preds_bin.view(-1).cpu().numpy(), average='binary')
    pixel_acc = (preds_bin == masks).float().mean().item()
    return iou, pixel_acc

# Run inference
ious, pixel_accuracies = [], []

with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc="Testing"):
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = torch.sigmoid(model(imgs))
        iou, acc = compute_metrics(preds, masks)
        ious.append(iou)
        pixel_accuracies.append(acc)

print(f"\nTest IoU: {np.mean(ious):.4f}")
print(f"Test Pixel Accuracy: {np.mean(pixel_accuracies):.4f}")

# Visualize a few predictions
for i in range(3):
    img, mask = test_dataset[i]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).cuda()))
        pred_bin = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(0), cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_bin, cmap='gray')
    plt.title("Predicted Mask")
    plt.tight_layout()
    plt.show()
