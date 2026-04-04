"""
(1.d) Web app: upload an image; model predicts B vs 0 vs E and shows probabilities.

Run after training:  python -m streamlit run app.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from image_preprocess import apply_standardization, image_to_feature_vector
from letter_patterns import CLASS_NAMES
from model import letter_mlp_from_checkpoint

# Default: wider net + improved training (higher accuracy than 64–3–3)
DEFAULT_CKPT = os.path.join("outputs", "model_h32.pt")


@st.cache_resource
def load_model(
    ckpt_path: str, device_str: str
) -> tuple[object, torch.device, np.ndarray | None, np.ndarray | None]:
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = letter_mlp_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    mu = ckpt.get("mu")
    std = ckpt.get("std")
    return model, device, mu, std


def main() -> None:
    st.set_page_config(page_title="Letter classifier (B / 0 / E)", layout="centered")
    st.title("TFML Assignment 2 — Letter prediction (B, 0, E)")
    st.markdown(
        "Upload a **clear** image of a capital **B**, digit **0**, or **E** on a light background. "
        "The model uses an **8×8** grayscale encoding (same convention as training: dark ≈ −1, bright ≈ +1)."
    )

    ckpt = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CKPT)
    device_opt = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

    if not os.path.isfile(ckpt):
        st.error(
            f"Checkpoint not found at `{ckpt}`. Train first, e.g. "
            "`python train.py --hidden 32` from the project folder."
        )
        return

    model, device, mu, std = load_model(ckpt, device_opt)
    up = st.file_uploader("Image (PNG/JPG)", type=["png", "jpg", "jpeg", "webp", "bmp"])

    if up is not None:
        pil = Image.open(up)
        st.image(pil, caption="Uploaded", use_container_width=True)
        vec = image_to_feature_vector(pil)
        vec = apply_standardization(vec, mu, std)
        x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(prob))
        st.subheader("Prediction")
        st.write(f"**Class:** {CLASS_NAMES[pred]}")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.bar(list(CLASS_NAMES), prob, color=["#2c7bb6", "#fdae61", "#d7191c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("8×8 preview (grayscale rescaled)"):
            g = (vec.reshape(8, 8) + 1) / 2
            st.image((g * 255).astype(np.uint8), width=200)


if __name__ == "__main__":
    main()
