import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 1. Define the Robust Augmentation Pipeline
def get_train_transforms():
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(height=224, width=224),
            # 80% Center Crop simulation (224 * 0.8 ≈ 179)
            A.OneOf(
                [
                    A.CenterCrop(height=179, width=179),
                    A.Downscale(scale_min=0.25, scale_max=0.5),  # Thumbnail generation
                ],
                p=0.5,
            ),
            A.ImageCompression(quality_range=(30, 90), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.0), p=0.3),
            A.GaussNoise(var_limit=(0.02 * 255, 0.10 * 255), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Resize(224, 224),  # Ensure final size matches model expectations
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class AIGCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# 2. Model Architecture
def build_model():
    # Load ConvNeXt-Tiny and freeze the feature extractor
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=1)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final classification head
    for param in model.head.parameters():
        param.requires_grad = True

    return model


# 3. Training Loop (Simplified for Hackathon)
def train_model(train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "aigc_detector.pth")
    return model
