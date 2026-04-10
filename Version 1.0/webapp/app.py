"""
app.py
======
Part 4: Flask web application for B / 0 / E character recognition.

Routes
------
GET  /           — serves the main UI
POST /predict    — accepts an uploaded image file, returns JSON prediction
POST /predict_canvas — accepts base64 canvas PNG, returns JSON prediction
"""

import base64
import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

import sys
ROOT        = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model import NeuralNetwork

MODEL_PATH  = PROJECT_ROOT / "models" / "net_64_3_3"
CLASS_NAMES = ["B", "0", "E"]

app   = Flask(__name__, template_folder="templates", static_folder="static")
model = NeuralNetwork.load(str(MODEL_PATH))


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Convert any PIL image to a normalised (1, 64) float32 array.
      1. Grayscale
      2. Resize to 8×8 with high-quality downsampling
      3. Threshold to binary {-1.0, +1.0}  (dark pixel → -1, light → +1)
    """
    img = pil_img.convert("L").resize((8, 8), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.where(arr > 127, 1.0, -1.0).reshape(1, 64)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a multipart image file upload and return prediction JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        img = Image.open(request.files["image"])
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    x     = preprocess_image(img)
    probs = model.predict_proba(x)[0]
    idx   = int(np.argmax(probs))

    return jsonify({
        "prediction":  CLASS_NAMES[idx],
        "confidences": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(3)},
        "pixel_grid":  x.reshape(8, 8).tolist(),
    })


@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """Accept a base64-encoded PNG from the canvas and return prediction JSON."""
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No canvas data received"}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
    except Exception:
        return jsonify({"error": "Invalid canvas data"}), 400

    x     = preprocess_image(img)
    probs = model.predict_proba(x)[0]
    idx   = int(np.argmax(probs))

    return jsonify({
        "prediction":  CLASS_NAMES[idx],
        "confidences": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(3)},
        "pixel_grid":  x.reshape(8, 8).tolist(),
    })


if __name__ == "__main__":
    app.run(debug=True)