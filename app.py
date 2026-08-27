import os
import sqlite3
import cv2
import torch
import timm
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from albumentations.pytorch import ToTensorV2
import albumentations as A

app = Flask(__name__)
CORS(app)


# Initialize SQLite database with WAL for concurrency protections
def init_db():
    conn = sqlite3.connect("inference_logs.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            ai_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

# Load model globally to keep it in memory
device = torch.device("cpu")
model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
model.load_state_dict(torch.load("aigc_detector.pth", map_location=device))
model.eval()

transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename

    # Read image directly from memory
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        tensor = transform(image=image)["image"].unsqueeze(0)
        prob = torch.sigmoid(model(tensor)).item()

    # Log to DB
    conn = sqlite3.connect("inference_logs.db")
    conn.execute(
        "INSERT INTO logs (filename, ai_score) VALUES (?, ?)", (filename, prob)
    )
    conn.commit()
    conn.close()

    return jsonify({"filename": filename, "ai_probability": round(prob, 4)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
