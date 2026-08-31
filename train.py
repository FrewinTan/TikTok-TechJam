import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset


# 1. Enhanced Real-World Degradation Pipeline
def get_train_transforms():
    return A.Compose(
        [  # type: ignore
            A.Resize(height=224, width=224),
            A.ImageCompression(quality_range=(25, 95), p=0.7),
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 2.5), p=0.5),
            A.GaussNoise(var_limit=(0.01 * 255.0, 0.12 * 255.0), p=0.4),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, p=0.4),
            A.CoarseDropout(max_holes=10, max_height=28, max_width=28, p=0.4),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


# 2. Balanced Streaming Dataset Class (Scaled to 10k samples)
class BalancedHFAIGCDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, max_samples=10000):
        real_samples = []
        fake_samples = []

        print(
            f"Balancing dataset distribution targeting {max_samples} total samples..."
        )
        for idx, item in enumerate(hf_dataset):
            if idx >= 40000:  # Expanded search window for larger sample size
                break

            raw_label = item["label"]
            binary_label = 0 if raw_label == 0 else 1

            if binary_label == 0 and len(real_samples) < max_samples // 2:
                real_samples.append(item)
            elif binary_label == 1 and len(fake_samples) < max_samples // 2:
                fake_samples.append(item)

            if (idx + 1) % 2000 == 0:
                print(
                    f"Scanned {idx + 1} items... Collected: {len(real_samples)} Real, {len(fake_samples)} Fake"
                )

            if (
                len(real_samples) >= max_samples // 2
                and len(fake_samples) >= max_samples // 2
            ):
                break

        self.samples = real_samples + fake_samples
        np.random.shuffle(self.samples)
        print(
            f"Dataset Ready: {len(real_samples)} Real, {len(fake_samples)} Fake (Total: {len(self.samples)})"
        )

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = np.array(item["image"].convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        raw_label = item["label"]
        binary_label = 0 if raw_label == 0 else 1
        label = torch.tensor(binary_label, dtype=torch.float32)
        return image, label


# 3. Main Training Execution with Scheduler
if __name__ == "__main__":
    print("Connecting to SID_Set stream...")
    hf_data = load_dataset("saberzl/SID_Set", split="train", streaming=True)

    # Increased to 10,000 samples for high-accuracy convergence
    dataset = BalancedHFAIGCDataset(
        hf_data, transform=get_train_transforms(), max_samples=10000
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training on device: {device}")

    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=1)

    # Phase 1: Freeze backbone, train head for 1 epoch
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)

    print("Starting optimized training loop...")

    for epoch in range(epochs):
        # Phase 2: Unfreeze entire backbone at epoch 1 with Cosine Annealing
        if epoch == 1:
            print("\n--- Unfreezing backbone for deep fine-tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=5e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - 1
            )

        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        if epoch >= 1:
            scheduler.step()  # Update learning rate smoothly

        print(
            f"Epoch {epoch+1} Completed | Average Loss: {running_loss/len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "aigc_detector.pth")
    print("Retraining complete! High-accuracy model saved as aigc_detector.pth")
