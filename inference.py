import argparse
import json
import os
import glob
import torch
import cv2
import timm
from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_inference_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="AIGC Detection Inference")
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory containing images"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    # Note: Ensure you have trained and saved 'aigc_detector.pth' before running
    model.load_state_dict(torch.load("aigc_detector.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = get_inference_transform()
    results = []

    # Process all images in directory
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))

    with torch.no_grad():
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)

            logits = model(tensor)
            # Apply sigmoid to get probability between 0 and 1
            prob = torch.sigmoid(logits).item()

            results.append({"image_path": img_path, "pred": round(prob, 4)})

    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processed {len(results)} images. Results saved to predictions.json.")


if __name__ == "__main__":
    main()
