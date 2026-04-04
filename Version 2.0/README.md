# TFML Assignment 2 — Synthetic B / 0 / E classifier

## Setup

```text
cd "Assignment 2"
pip install -r requirements.txt
python train.py --hidden 32 --out-dir outputs
python experiments.py --out-dir outputs
```

For **strict 1.a (64–3–3)**, use `--hidden 3` (accuracy is lower than wider hidden layers under heavy noise).

## Accuracy improvements (implemented)

- **Optimizer:** AdamW with weight decay (L2-style regularization).
- **Loss:** label smoothing (reduces overconfident fits on noisy labels).
- **Schedule:** OneCycle LR (or `--scheduler cosine` / `none`).
- **Training stability:** gradient clipping; **restore best validation weights** (not necessarily the last epoch).
- **Architecture:** optional **LayerNorm** on the hidden layer, **GELU** (or `--activation relu|tanh|silu`), **dropout** scaled by hidden width (`--dropout -1` = automatic).
- **Inputs:** per-feature standardization fit on the training split only (saved in the checkpoint for the web app).

## Web app (1.d)

After `outputs/model_h32.pt` exists (or point the sidebar to any `outputs/model_h*.pt`):

```text
streamlit run app.py
```

Upload a **high-contrast** image (dark letter on light background works best). The app resizes to 8×8 and applies the **same standardization** (train-split mean/std) as training.

## Files

| File | Role |
|------|------|
| `letter_patterns.py` | Canonical 8×8 patterns (−1 black, +1 white) for B, 0, E |
| `dataset.py` | Dataset **D**: 300 samples, uniform noise U[−5,5] per pixel |
| `model.py` | 64–X–3 MLP with biases; optional LayerNorm, dropout, GELU/etc. |
| `train.py` | Train, save checkpoint + weight figures |
| `visualize_weights.py` | (1.b) 8×8 input→hidden; (1.c) hidden→output heatmap + bars |
| `experiments.py` | (1.e) sweep over X and sample size; saves PNG + `experiment_report.txt` |
| `app.py` | Streamlit upload + prediction |
| `image_preprocess.py` | Resize 8×8, map pixels to [−1,+1], optional standardization |

## Written answers (for your report)

### (1.b) Interpretation — input→hidden weights as 8×8 images

Each hidden unit’s incoming weights form a 64-vector; reshaping to 8×8 shows a **spatial sensitivity map** over the grid. After training, you typically see **positive and negative regions** that roughly align with stroke vs background areas, because the three classes differ in which pixels are dark (−1) vs light (+1). With only three hidden units, each unit acts like a coarse **feature detector** (a learned combination of pixel contrasts), not a perfect template of a letter.

### (1.c) Interpretation — hidden→output weights

The matrix **W₂** has shape **3 × X**: row *k* contains weights from all hidden units to output *k* (logit for class B, 0, or E). Each row answers: “**how should this class combine the hidden features?**” The bar charts per output show which hidden units **raise or lower** that class score. The output biases shift logits when hidden activations are weak. Together, the second layer is a **linear classifier in hidden-feature space**.

### (1.e) Architecture and sample complexity

See `outputs/exp_architecture_vs_hidden.png` and `outputs/exp_sample_complexity.png`, and numeric summaries in `outputs/experiment_report.txt`. Under heavy additive noise, **wider hidden layers (larger X)** usually improve validation accuracy up to a point (diminishing returns / variance). **More training samples** generally help, but curves can be noisy because each sample size uses newly drawn noise; averaging over seeds (as in the script) reduces that noise.

## Note on training

Inputs are **per-feature standardized** using the **training split only** (mean/std saved in the checkpoint) so optimization remains stable when noise has large magnitude. This does not change the letter labels or the noise model for generating **D**; it only rescales inputs to the network.
