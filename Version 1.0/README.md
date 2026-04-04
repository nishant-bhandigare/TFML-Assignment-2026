# Character Recognition — Neural Network Project
### CS6302E · Theoretical Foundations of Machine Learning · NIT Calicut

A complete system that recognises the characters **B**, **0** (zero), and **E** from
8×8 pixel images using a fully-connected feedforward neural network built from
scratch with NumPy.

---

## Project Structure

```
char_recognition/
├── data_generation.py      # Part 1 — templates, noise, dataset creation
├── model.py                # Part 2 — 64-H-3 network, Adam, backprop
├── train.py                # Part 2 — training loop, plots, confusion matrix
├── visualize_weights.py    # Part 3 — W1 heatmaps, W2 bar charts
├── architecture_search.py  # Part 5 — hidden-size search, sample complexity
├── requirements.txt        # Python dependencies
├── notebooks/
│   └── character_recognition.ipynb   # Optional end-to-end notebook
│
├── webapp/
│   ├── app.py              # Part 4 — Flask backend
│   ├── static/             # (optional assets)
│   └── templates/
│       └── index.html      # Part 4 — frontend UI
│
├── data/                   # Generated dataset (.npy files)
│   ├── X.npy
│   ├── y.npy
│   └── labels.npy
│
├── models/                 # Saved model weights
│   └── net_64_3_3.npz
│
├── plots/                  # All generated plots
│   ├── templates.png
│   ├── noisy_samples.png
│   ├── pixel_distribution.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── weights_input_hidden.png
│   ├── weights_overlay.png
│   ├── weights_hidden_output.png
│   ├── biases.png
│   ├── arch_search_accuracy.png
│   ├── arch_search_loss_curves.png
│   └── sample_complexity.png
│
└── README.md
```

---

## Requirements

- Python 3.9+
- numpy
- matplotlib
- seaborn
- flask
- pillow

Install all dependencies:

```bash
pip install -r requirements.txt
```

(or `pip install numpy matplotlib seaborn flask pillow`)

---

## Quick Start — Run Everything

```bash
# 1. Generate the dataset
python data_generation.py

# 2. Train the 64-3-3 network
python train.py

# 3. Visualise weights
python visualize_weights.py

# 4. Launch the web app (run from webapp/ so templates resolve correctly)
cd webapp
python app.py
# → Open http://127.0.0.1:5000  (use: python app.py --port 5001 if 5000 is busy)

# 5. Architecture search (takes a few minutes)
cd ..
python architecture_search.py
```

---

## Part 1 — Dataset Generation

`data_generation.py` creates three 8×8 binary templates for **B**, **0**, **E**
(pixel values −1.0 = black, +1.0 = white). It then generates 100 noisy copies of
each character by adding independent uniform noise in [−5.0, +5.0] to every
pixel, producing a 300-sample dataset saved to `data/`.

---

## Part 2 — Neural Network (64-3-3)

`model.py` implements the network from scratch with NumPy:

| Layer | Units | Activation |
|-------|-------|------------|
| Input | 64 | — |
| Hidden | 3 (default) | **tanh** |
| Output | 3 | **softmax** |

- Loss: Cross-Entropy
- Optimizer: Adam (β₁=0.9, β₂=0.999, lr=0.001)
- Weights: Xavier/Glorot initialisation
- Early stopping: patience=300 epochs

`train.py` runs the training loop with an 80/20 stratified split and saves the
best model (by validation loss) to `models/net_64_3_3.npz`.

---

## Part 3 — Weight Visualisation

`visualize_weights.py` produces four plots:

1. **weights_input_hidden.png** — Each hidden unit's 64 input weights reshaped
   to 8×8, shown as a RdBu diverging heatmap.
2. **weights_overlay.png** — Character templates alongside the weight maps for
   direct comparison.
3. **weights_hidden_output.png** — Grouped bar chart of W2 (3×3) plus an
   annotated heatmap matrix.
4. **biases.png** — Bias values for both layers.

---

## Part 4 — Web Application

`webapp/app.py` is a Flask server with three endpoints:

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves the main UI |
| `/predict` | POST | Accepts image (file or base64), returns prediction JSON |
| `/model_info` | GET | Returns architecture metadata |
| `/template/<char>` | GET | Returns clean template pixels for B/0/E |

### UI Features
- **Draw tab** — freehand canvas with adjustable brush; right-click to erase
- **Upload tab** — drag-and-drop or file picker
- Confidence bars for all three classes
- 8×8 pixel preview of the preprocessed input fed to the network
- Reference template thumbnails (click to load into canvas)
- Live model info strip

### Image Preprocessing Pipeline
1. Convert to grayscale
2. Resize to 8×8 (LANCZOS)
3. Normalise: `(pixel / 127.5) − 1.0` → [−1.0, +1.0]
4. Flatten to 64-dim vector

---

## Part 5 — Architecture Search

`architecture_search.py` trains a `64-X-3` network for each hidden size
`X ∈ {1, 2, 3, 4, 5, 8, 10, 16, 32}` and records train/val accuracy and loss.
It also performs a **sample complexity analysis** by varying training-set size
per class across `{10, 20, 50, 100, 200}` for each architecture.

Generated plots:
- `arch_search_accuracy.png` — Accuracy vs hidden layer size
- `arch_search_loss_curves.png` — Loss curves for every architecture
- `sample_complexity.png` — Accuracy vs training-set size per architecture

---

## Notes on AI-Generated Code

All code in this project was developed with AI assistance and has been reviewed
and commented throughout. Key design decisions are documented inline:

- `model.py` — Xavier initialisation rationale, softmax numerical stability,
  Adam bias-correction derivation
- `train.py` — stratified split, early-stopping logic
- `visualize_weights.py` — TwoSlopeNorm for zero-centred diverging colormap
- `webapp/app.py` — preprocessing pipeline, dual-format endpoint

---

## Authors

NIT Calicut · CS6302E Group Submission · Winter 2025–26
