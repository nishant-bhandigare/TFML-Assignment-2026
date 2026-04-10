"""Flask web app for B/0/E prediction."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model.keras"
STATS_PATH = ROOT / "data" / "processed" / "norm_stats.npy"
CLASS_NAMES = ["B", "0", "E"]

app = Flask(__name__, template_folder="templates", static_folder="static")
model = tf.keras.models.load_model(MODEL_PATH)
mean, std = np.load(STATS_PATH)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L").resize((8, 8), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.where(arr > 127, 1.0, -1.0).reshape(1, 64)
    arr = (arr - float(mean)) / (float(std) + 1e-8)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    img = Image.open(request.files["image"])
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidences": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
    })


if __name__ == "__main__":
    app.run(debug=True)
