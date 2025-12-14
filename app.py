import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import gdown

# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi_indonesian_foods.csv"

# Google Drive FILE ID (model)
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"

IMG_SIZE = (128, 128)

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# LOAD LABELS
# =========================
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# LOAD TKPI DATA
# =========================
@st.cache_data
def load_tkpi():
    return pd.read_csv(TKPI_PATH)

tkpi_df = load_tkpi()

# =========================
# UI
# =========================
st.title("🍱 Food Image Recognition AI")
st.write("Upload a food image and let the AI predict the category.")

uploaded_file = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # =========================
    # PREPROCESS
    # =========================
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICT
    # =========================
    preds = model.predict(img_array)
    probs = tf.nn.softmax(preds[0]).numpy()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    predicted_label = class_names[class_id]

    # =========================
    # OUTPUT: AI PREDICTION
    # =========================
    st.subheader("🔍 Prediction")
    st.write(f"**Label:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # =========================
    # OUTPUT: TKPI DATA
    # =========================
    st.subheader("🍽 Informasi Gizi (TKPI 2017)")

    food_info = tkpi_df[tkpi_df["label"] == predicted_label]

    if food_info.empty:
        st.info("Data gizi tidak tersedia untuk makanan ini.")
    else:
        st.table(food_info.drop(columns=["label"]))

    st.caption("📚 Sumber data gizi: TKPI 2017 (per 100 gram)")

    # =========================
    # CONFIDENCE BREAKDOWN
    # =========================
    with st.expander("📊 Confidence Breakdown"):
        for i, c in enumerate(class_names):
            st.write(f"{c}: {probs[i]:.3f}")
