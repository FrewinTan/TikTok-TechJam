import os
import argparse
import json
import cv2
import numpy as np
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_inference_transforms():
    return A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


# Test-Time Augmentation (TTA) to handle variable judge conditions smoothly
def predict_with_tta(model, image, device, transform):
    variants = [
        image,  # Original
        cv2.resize(image, (200, 200)),  # Slight scale down variation
        cv2.flip(image, 1),  # Horizontally flipped variation
    ]

    probs = []
    for var in variants:
        tensor = transform(image=var)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs.append(torch.sigmoid(logits).item())

    # Return average probability across all transformed variations
    return sum(probs) / len(probs)


def main():
    parser = argparse.ArgumentParser(description="Run AIGC Batch Inference with TTA")
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Path to input image directory"
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running inference on device: {device}")

    print("Loading model...")
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load("aigc_detector.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = get_inference_transforms()
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    results = []

    print(f"Scanning directory: {args.img_dir}")
    for filename in os.listdir(args.img_dir):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(args.img_dir, filename)

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Predict using TTA robustness engine
            prob = predict_with_tta(model, image, device, transform)

            # Calibrated 0.65 threshold
            is_fake = prob >= 0.65

            results.append(
                {
                    "image_path": img_path,
                    "pred": round(prob, 4),
                    "prediction": "AI-Generated" if is_fake else "Authentic",
                }
            )
            print(
                f"Processed: {filename} -> Score: {prob:.4f} ({'AI-Generated' if is_fake else 'Authentic'})"
            )

    # Output mandatory JSON deliverable for judges
    output_file = "predictions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nInference complete! Results successfully saved to {output_file}")


if __name__ == "__main__":
    main()
