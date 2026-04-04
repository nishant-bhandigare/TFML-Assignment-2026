# Character Recognition — Setup & Run Guide
### CS6302E · NIT Calicut · Neural Network Project

---

## Directory Structure

Build this exact folder layout before placing any files:

```
char_recognition/
│
├── data_generation.py          ← Part 1  (download from outputs)
├── model.py                    ← Part 2  (download from outputs)
├── train.py                    ← Part 2  (download from outputs)
├── visualize_weights.py        ← Part 3  (download from outputs)
├── architecture_search.py      ← Part 5  (download from outputs)
├── README.md                   ← (download from outputs)
│
├── webapp/
│   ├── app.py                  ← Part 4  (download from outputs)
│   └── templates/
│       └── index.html          ← Part 4  (download from outputs)
│
├── data/                       ← auto-created by data_generation.py
│   ├── X.npy
│   ├── y.npy
│   └── labels.npy
│
├── models/                     ← auto-created by train.py
│   ├── net_64_3_3.npz
│   └── net_64_3_3_best.npz
│
└── plots/                      ← auto-created by each script
    ├── templates.png
    ├── noisy_samples.png
    ├── pixel_distribution.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── weights_input_hidden.png
    ├── weights_overlay.png
    ├── weights_hidden_output.png
    ├── biases.png
    ├── arch_search_accuracy.png
    ├── arch_search_loss_curves.png
    ├── sample_complexity.png
    └── acc_vs_X_by_samples.png
```

> The `data/`, `models/`, and `plots/` folders are created automatically
> when you run the scripts. You do NOT need to create them manually.

---

## Step 1 — Place the Downloaded Files

From the files Claude gave you, place each one as follows:

| Downloaded file          | Goes to                                  |
|--------------------------|------------------------------------------|
| `data_generation.py`     | `char_recognition/data_generation.py`   |
| `model.py`               | `char_recognition/model.py`             |
| `train.py`               | `char_recognition/train.py`             |
| `visualize_weights.py`   | `char_recognition/visualize_weights.py` |
| `architecture_search.py` | `char_recognition/architecture_search.py` |
| `app.py`                 | `char_recognition/webapp/app.py`        |
| `index.html`             | `char_recognition/webapp/templates/index.html` |
| `README.md`              | `char_recognition/README.md`            |

---

## Step 2 — Requirements

Python **3.9 or higher** is required.

Install all dependencies in one command:

```bash
pip install -r requirements.txt
```

No PyTorch or TensorFlow needed — the neural network is pure NumPy.

---

## Step 3 — Run Each Part in Order

**Always run every command from inside the `char_recognition/` folder:**

```bash
cd char_recognition
```

---

### Part 1 — Generate the Dataset

```bash
python data_generation.py
```

**What it does:**
- Creates the 8×8 pixel templates for B, 0, E
- Generates 300 noisy training samples (100 per class)
- Saves `data/X.npy`, `data/y.npy`, `data/labels.npy`
- Saves 3 plots to `plots/`

**Expected output:**
```
Class 'B' (0): 100 samples generated.
Class '0' (1): 100 samples generated.
Class 'E' (2): 100 samples generated.
Dataset saved to 'data/'
```

---

### Part 2 — Train the Neural Network

```bash
python train.py
```

**What it does:**
- Loads the dataset from `data/`
- Builds the 64-3-3 network with Xavier initialisation
- Trains with Adam optimiser (max 3000 epochs, early stopping)
- Saves the best model to `models/net_64_3_3.npz`
- Saves `plots/training_curves.png` and `plots/confusion_matrix.png`

**Expected output (abridged):**
```
Epoch    1 | train loss 1.43  acc 32.1% | val loss 1.55  acc 36.7%
Epoch  200 | train loss 0.20  acc 92.9% | val loss 0.66  acc 76.7%
Early stop at epoch 520
Train accuracy : 92.92%
Val   accuracy : 81.67%
Model weights saved → models/net_64_3_3.npz
```

---

### Part 3 — Visualise Weights

```bash
python visualize_weights.py
```

**What it does:**
- Loads the saved model from `models/net_64_3_3.npz`
- Generates 4 plots showing input→hidden and hidden→output weights
- Prints a written interpretation to the terminal

**Saves:**
- `plots/weights_input_hidden.png`
- `plots/weights_overlay.png`
- `plots/weights_hidden_output.png`
- `plots/biases.png`

---

### Part 4 — Launch the Web Application

```bash
cd webapp
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

**Features:**
- **Draw tab** — draw a character on the canvas with your mouse
  - Left-click = draw (white)
  - Right-click = erase (black)
  - Adjust brush size with the slider
- **Upload tab** — drag-and-drop or click to upload any image (PNG, JPG, BMP)
- **Classify button** — runs the model and shows:
  - The predicted class (B, 0, or E)
  - Confidence percentages for all three classes
  - The 8×8 pixel preview of what the network actually sees
- **Reference templates** at the bottom — click any to load it into the canvas

To stop the server: press `Ctrl+C` in the terminal.

> **Note:** Run `python app.py` from inside the `webapp/` folder,
> or it won't find the model file via the relative path.

---

### Part 5 — Architecture Search

Go back to the `char_recognition/` root first:

```bash
cd ..          # if you're still inside webapp/
python architecture_search.py
```

**What it does:**
- Trains 64-X-3 networks for X ∈ {1, 2, 3, 4, 5, 8, 10, 16, 32}
- Records train/val accuracy and loss for each
- Runs sample complexity analysis across {10, 20, 50, 100, 200} samples/class
- Prints a full summary table and written justification
- Saves the best architecture model

**This takes 3–5 minutes to complete.**

**Saves:**
- `plots/arch_search_accuracy.png`
- `plots/arch_search_loss_curves.png`
- `plots/sample_complexity.png`
- `plots/acc_vs_X_by_samples.png`
- `models/net_64_3_3_best.npz`

---

## Full Run — All Parts in Sequence

```bash
cd char_recognition

python data_generation.py
python train.py
python visualize_weights.py
python architecture_search.py

# Then launch the web app:
cd webapp
python app.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install flask` |
| `ModuleNotFoundError: No module named 'PIL'` | Run `pip install pillow` |
| `FileNotFoundError: data/X.npy` | Run `python data_generation.py` first |
| `FileNotFoundError: models/net_64_3_3.npz` | Run `python train.py` first |
| `Address already in use` (Flask) | Another process is using port 5000. Kill it or run `python app.py` with `--port 5001` |
| Plots show but don't save | Make sure you're running from inside `char_recognition/`, not a subfolder |
| Web app can't find model | Make sure you're running `python app.py` from inside `webapp/`, not from root |

---

## Python Version Check

```bash
python --version   # must be 3.9 or higher
```

If your system uses `python3` instead of `python`:

```bash
python3 data_generation.py
python3 train.py
# etc.
```
