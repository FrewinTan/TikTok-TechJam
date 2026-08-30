import os
import random
import json
import itertools
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from PIL import Image

# Config

HF_DATASET = "saberzl/SID_Set"
MODEL_NAME = "convnext_tiny"

OUTPUT_PATH = "aigc_detector_final.pth"
RESUME_PATH = "resume_checkpoint.pth"
METRICS_PATH = "training_metrics.json"

SEED = 42
IMG_SIZE = 224

BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4

STEPS_PER_EPOCH = 1000

VAL_SAMPLES = 2000

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# Seed

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Transforms

def get_train_transforms():
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),

            A.OneOf(
                [
                    A.ImageCompression(quality_range=(30, 90), p=1.0),
                    A.GaussianBlur(
                        blur_limit=(3, 7),
                        sigma_limit=(0.5, 2.0),
                        p=1.0,
                    ),
                    A.Downscale(
                        scale_range=(0.25, 0.50),
                        p=1.0,
                    ),
                ],
                p=0.7,
            ),

            A.GaussNoise(std_range=(0.02, 0.10), p=0.35),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.35,
            ),

            A.OneOf(
                [
                    A.CenterCrop(height=179, width=179, p=1.0),
                    A.NoOp(p=1.0),
                ],
                p=0.3,
            ),

            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

def get_val_transforms():
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

# Iterable Streaming Dataset

class StreamingSIDDataset(IterableDataset):
    def __init__(
        self,
        split="train",
        transform=None,
        shuffle_buffer=1000,
        seed=SEED,
        skip=0,
        take=None,
    ):
        self.split = split
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip = skip
        self.take = take

    def parse_sample(self, sample):
        image = sample["image"]
        label = float(sample["label"])

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = np.array(image)

            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            if image.shape[-1] == 4:
                image = image[:, :, :3]

            image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(label, dtype=torch.float32)
        return image, label

    def __iter__(self):
        ds = load_dataset(
            HF_DATASET,
            split=self.split,
            streaming=True,
        )

        # Shuffle in streaming mode with a limited buffer.
        # Avoids downloading the whole dataset first.
        ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        if self.skip > 0:
            ds = ds.skip(self.skip)

        if self.take is not None:
            ds = ds.take(self.take)

        for sample in ds:
            yield self.parse_sample(sample)

# Model

def build_model():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=1,
    )

    # Freeze all first for speed.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final stage + head for better performance while still fast.
    if hasattr(model, "stages"):
        for param in model.stages[-1].parameters():
            param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

    return model

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {MODEL_NAME}")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    if total >= 2_000_000_000:
        raise ValueError("Model exceeds 2B parameter limit.")

# Metrics

def compute_binary_metrics(y_true, y_pred):
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def evaluate(model, val_loader, device, max_batches=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().view(-1).tolist()
            labs = labels.long().cpu().view(-1).tolist()

            y_pred.extend(preds)
            y_true.extend(labs)

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(y_true) // BATCH_SIZE)
    return metrics

# Training

def save_resume_checkpoint(epoch, model, optimizer, best_f1):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": best_f1,
        },
        RESUME_PATH,
    )

def load_resume_checkpoint(model, optimizer, device):
    if not os.path.exists(RESUME_PATH):
        return 0, -1.0

    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_f1 = checkpoint.get("best_f1", -1.0)

    print(f"Resumed from epoch {start_epoch}. Best F1 so far: {best_f1:.4f}")
    return start_epoch, best_f1

def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Using Hugging Face streaming mode.")
    print("This avoids storing the full SID_Set locally.")
    print(f"Dataset: {HF_DATASET}")

    train_dataset = StreamingSIDDataset(
        split="train",
        transform=get_train_transforms(),
        shuffle_buffer=2000,
        seed=SEED,
        skip=0,
        take=None,
    )

    val_dataset = StreamingSIDDataset(
        split="train",
        transform=get_val_transforms(),
        shuffle_buffer=1000,
        seed=SEED + 999,
        skip=10_000,
        take=VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model().to(device)
    count_params(model)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4,
    )

    start_epoch, best_f1 = load_resume_checkpoint(model, optimizer, device)

    history = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        step_count = 0

        print(f"\nStarting epoch {epoch + 1}/{EPOCHS}")

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Step {step} | "
                    f"Train Loss: {running_loss / step_count:.4f}"
                )

            if STEPS_PER_EPOCH is not None and step >= STEPS_PER_EPOCH:
                break

        train_loss = running_loss / max(1, step_count)

        print("Running validation...")
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(log)

        with open(METRICS_PATH, "w") as f:
            json.dump(history, f, indent=4)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"Saved best model to {OUTPUT_PATH}")

        save_resume_checkpoint(epoch + 1, model, optimizer, best_f1)
        print(f"Saved resume checkpoint to {RESUME_PATH}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining complete.")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best model saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    train()
