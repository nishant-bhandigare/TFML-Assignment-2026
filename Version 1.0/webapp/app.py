"""
webapp/app.py
=============
Part 4: Flask web application for character recognition.

Routes:
  GET  /            → serves the main UI page
  POST /predict     → accepts base64 image OR uploaded file,
                      preprocesses → 8×8, normalises → [−1, +1],
                      runs inference, returns JSON with class + confidences
  GET  /model_info  → returns architecture details and weight stats as JSON
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Path setup: project root (parent of webapp/) for imports and model file
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model import NeuralNetwork  # noqa: E402

# Load model once at startup (64-3-3 weights from train.py)
MODEL_PATH = os.path.join(ROOT, "models", "net_64_3_3.npz")
net = NeuralNetwork.load(MODEL_PATH)

CLASS_NAMES = ["B", "0", "E"]
CLASS_DESCRIPTIONS = {
    "B": "Capital letter B — two bumps on the right, vertical stroke on the left",
    "0": "Zero — smooth oval / ellipse, no corners",
    "E": "Capital letter E — three horizontal bars extending from a vertical stroke",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to the model's expected 64-dim input vector.

    Pipeline:
      1. Convert to grayscale (L mode)
      2. Resize to 8×8 using LANCZOS resampling
      3. Convert pixel values [0, 255] → [−1.0, +1.0]
         formula:  norm = (pixel / 127.5) − 1.0
      4. Flatten to shape (64,)
    """
    img_gray = img.convert("L")
    img_small = img_gray.resize((8, 8), Image.LANCZOS)
    arr = np.array(img_small, dtype=np.float64)
    arr_norm = (arr / 127.5) - 1.0
    return arr_norm.flatten()


def predict(pixel_vector: np.ndarray) -> dict:
    """Run inference and return structured prediction result."""
    X = pixel_vector.reshape(1, -1)
    probs = net.predict_proba(X)[0]
    idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[idx],
        "predicted_index": idx,
        "description": CLASS_DESCRIPTIONS[CLASS_NAMES[idx]],
        "confidences": {
            name: float(f"{p * 100:.2f}")
            for name, p in zip(CLASS_NAMES, probs)
        },
        "pixel_preview": pixel_vector.tolist(),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accept image in two formats:
      • multipart/form-data  with key 'image'   (file upload)
      • application/json     with key 'image_b64' (canvas base64 PNG)
    """
    try:
        img = None

        if "image" in request.files:
            f = request.files["image"]
            if f.filename == "":
                return jsonify({"error": "No file selected"}), 400
            img = Image.open(f.stream)

        elif request.is_json:
            data = request.get_json()
            b64str = data.get("image_b64", "")
            if not b64str:
                return jsonify({"error": "No image data provided"}), 400
            if "," in b64str:
                b64str = b64str.split(",", 1)[1]
            raw = base64.b64decode(b64str)
            img = Image.open(io.BytesIO(raw))

        else:
            return jsonify({"error": "Send image as file or base64 JSON"}), 400

        vec = preprocess_image(img)
        result = predict(vec)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info")
def model_info():
    """Return model architecture and weight statistics."""
    W1, W2 = net.W1, net.W2
    return jsonify({
        "architecture": f"64 → {net.hidden_dim} → {net.output_dim}",
        "total_params": int(W1.size + net.b1.size + W2.size + net.b2.size),
        "hidden_dim": net.hidden_dim,
        "activation": "tanh (hidden)  +  softmax (output)",
        "optimizer": "Adam",
        "loss": "Cross-Entropy",
        "W1_shape": list(W1.shape),
        "W2_shape": list(W2.shape),
        "W1_range": [float(f"{W1.min():.3f}"), float(f"{W1.max():.3f}")],
        "W2_range": [float(f"{W2.min():.3f}"), float(f"{W2.max():.3f}")],
        "classes": CLASS_NAMES,
    })


@app.route("/template/<char>")
def get_template(char: str):
    """Return the clean 8×8 template for a character as a flat list."""
    from data_generation import TEMPLATE_B, TEMPLATE_0, TEMPLATE_E

    tmpl_map = {"B": TEMPLATE_B, "0": TEMPLATE_0, "E": TEMPLATE_E}
    if char not in tmpl_map:
        return jsonify({"error": "Unknown class"}), 404
    return jsonify({
        "char": char,
        "pixels": tmpl_map[char].flatten().tolist(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character recognition web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print("  Character Recognition Web App")
    print(f"  Model : {net}")
    print(f"  Open  : http://{args.host}:{args.port}")
    print(f"{'=' * 50}\n")
    app.run(debug=True, host=args.host, port=args.port)
