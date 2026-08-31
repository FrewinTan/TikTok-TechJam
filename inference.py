import argparse
import json
import os
import glob
import torch
import cv2
import timm
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

MODEL_PATH = "aigc_detector.pth"
THRESHOLD = 0.8

def get_inference_transforms():
    return [
        A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.CenterCrop(224, 224),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.Resize(224, 224),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    ]

def main():
    parser = argparse.ArgumentParser(description="AIGC Detection Inference")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transforms = get_inference_transforms()
    results = []

    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))

    with torch.no_grad():
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            probs = []
            for transform in transforms:
                tensor = transform(image=image)["image"].unsqueeze(0).to(device)
                logits = model(tensor)
                prob = torch.sigmoid(logits).item()
                probs.append(prob)

            avg_prob = float(np.mean(probs))
            label = "ai" if avg_prob >= THRESHOLD else "real"

            results.append(
                {
                    "image_path": img_path,
                    "pred": round(avg_prob, 4),
                    "label": label
                }
            )

            print(f"{img_path} -> score={avg_prob:.4f}, label={label}")

    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processed {len(results)} images. Results saved to predictions.json.")

if __name__ == "__main__":
    main()
