import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi.csv"  # CSV file path

# Google Drive file IDs (for large files)
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"

# =========================
# FUNCTIONS
# =========================

# 1️⃣ Load CSV safely
@st.cache_data
def load_csv(path):
    """
    Load CSV safely with multiple encoding fallback
    """
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to read CSV: {path}. Unsupported encoding.")
    return pd.DataFrame()  # return empty DataFrame as fallback

# 2️⃣ Load model (download from Google Drive if not present)
@st.cache_resource
def load_model(path=MODEL_PATH, gdrive_id=None):
    if not os.path.exists(path):
        if gdrive_id is None:
            st.error("Model not found and no Google Drive ID provided!")
            return None
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, path, quiet=False)
    model = tf.keras.models.load_model(path)
    return model

# 3️⃣ Load labels
@st.cache_data
def load_labels(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except Exception as e:
        st.error(f"Failed to load labels: {e}")
        return []

# =========================
# LOAD DATA
# =========================
tkpi_df = load_csv(TKPI_PATH)
model = load_model(MODEL_PATH, GDRIVE_MODEL_ID)
labels = load_labels(LABELS_PATH)

# =========================
# STREAMLIT UI
# =========================
st.title("Food AI App")

st.subheader("TKPI Data Preview")
st.dataframe(tkpi_df.head())

st.subheader("Predict Food Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)] if labels else "Unknown"

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted: {predicted_label}")
