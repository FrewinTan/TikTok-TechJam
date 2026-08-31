***Project Overview***
This repository contains our prototype for detecting AI-Generated Content (AIGC). Current detection models often achieve high accuracy in lab environments but fail when images are subjected to social media redistribution (compression, cropping, filtering).

Our solution directly addresses the hackathon's problem statement by prioritising robustness over fragile lab-only accuracy.

Key Technical Highlights:

Model Selection: We utilize a pre-trained convnext_tiny backbone via the timm library. At roughly 28 million parameters, it strictly adheres to the TechJam's < 2B parameter limit while delivering state-of-the-art ConvNet feature extraction.
Real-World Degradation Pipeline: During training, we use Albumentations to heavily penalize pristine data. We simulate the exact real-world analogs outlined in the prompt (Social media re-encoding, Out-of-focus, Low-light sensor noise, Filter apps) using probabilistic JPEG Compression, Gaussian Blur, Gaussian Noise, and Color Jitter.
Two-Phase Fine-Tuning: We employ a stabilized training approach—initially freezing the backbone to calibrate the classification head, then unfreezing the entire network using a Cosine Annealing learning rate scheduler to learn deep AIGC artifacts.
Test-Time Augmentation (TTA): During inference, our script doesn't just evaluate a single image. It creates structural variants (e.g., resizing, flipping) and averages the confidence scores, drastically reducing false positives on edge cases.
Dataset Compliance: We trained on a balanced 10k sample subset streamed from Hugging Face (saberzl/SID_Set). We strictly avoided the prohibited validation subsets (COCO val2017 & DALL-E Advanced).


***Setup and Installation Instructions***
Prerequisites
Python 3.8+
Git
Hardware: A CUDA-enabled GPU or Apple Silicon (MPS) is recommended for training, though the scripts will fall back to CPU if necessary.

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install required dependencies:
pip install torch torchvision
pip install timm albumentations opencv-python datasets numpy


***Steps to Reproduce Results***
1. Training the Model
To reproduce our trained weights from scratch, run the training script. This script automatically connects to the Hugging Face dataset stream, balances the classes (5k Real / 5k Fake), applies our degradation augmentations, and trains the convnext_tiny model over 5 epochs.

python train.py
Note: Upon completion, this script will save the model weights as aigc_detector.pth in the root directory.

2. Running Inference (Judging Script)
As required by the hackathon guidelines, we have provided a script that takes a directory of images as input and outputs a standardized JSON file with confidence scores.

To run the evaluation:

python inference.py --img_dir test_images 
Output: The script will generate a file named predictions.json in the current directory. The output follows the format:


[
    {
        "image_path": "path",
        "pred": 0.8924,
        "prediction": "AI-Generated"
    },
    {
        "image_path": "path",
        "pred": 0.1205,
        "prediction": "Authentic"
    }
]
(Scores closer to 1.0 indicate a high likelihood of being AI-Generated).


***Reflections and Limitations***
Given the "hackathon-scale" compute and time limits, our prototype makes calculated trade-offs.

Limitations:

Global vs. Local Artifacts: convnext_tiny analyzes the image globally. While highly effective at detecting overall generative textures (like diffusion noise), it may struggle with images that are 95% authentic but contain small, localized AI-inpainting (e.g., an AI-generated face swapped onto a real body).
TTA Inference Overhead: Our use of Test-Time Augmentation (TTA) makes predictions significantly more robust to unexpected compression, but it means inference takes slightly longer per image compared to a single forward pass.
Unseen Modalities: Our training data heavily features diffusion models. Generative approaches relying on fundamentally different architectures (like certain older GANs) might bypass the specific high-frequency filters our model learned.
Future Improvements (Given more time):

Frequency Domain Analysis: We would implement Discrete Cosine Transform (DCT) preprocessing. AIGC often leaves invisible, repeating spectral artifacts in the frequency domain that are completely immune to standard color jittering or JPEG compression.
Explicit Thumbnail Generation: We would update our training pipeline to explicitly include a scale down -> scale up augmentation to perfectly mimic the "Thumbnail generation" constraint listed in the prompt.
Ensemble Approach: Combining convnext_tiny with a lightweight Vision Transformer (ViT) to capture both local convolutional features and global attention features, while still remaining under the 2B parameter cap.


***Team Member Contributions***
Frewin: Contributed to architectural design, implemented train.py with two-phase fine-tuning, and engineered the Albumentations degradation pipeline.

Jerry: Contributed to architectural design, developed inference.py, implemented Test-Time Augmentation logic, and handled dataset streaming optimization.

Yin Fan: Managed Devpost documentation, and recorded/edited the final demonstration video.
