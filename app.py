import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import gdown
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi_indonesian_foods.csv"

# Google Drive file ID for large model
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"

# =========================
# FUNCTIONS
# =========================

# 1️⃣ Load CSV safely
@st.cache_data
def load_csv(path):
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines='skip')
            return df
        except Exception:
            continue
    st.error(f"Failed to read CSV: {path}.")
    return pd.DataFrame()

# 2️⃣ Load Dense model (trained on features)
@st.cache_resource
def load_dense_model(path=MODEL_PATH, gdrive_id=None):
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

# 4️⃣ Feature extractor (CNN base)
@st.cache_resource
def load_cnn_base():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    return base_model

# =========================
# LOAD DATA & MODELS
# =========================
tkpi_df = load_csv(TKPI_PATH)
dense_model = load_dense_model(MODEL_PATH, GDRIVE_MODEL_ID)
labels = load_labels(LABELS_PATH)
cnn_base = load_cnn_base()  # feature extractor

# =========================
# STREAMLIT UI
# =========================
st.title("Food AI App")

# TKPI CSV preview
st.subheader("TKPI Data Preview")
st.dataframe(tkpi_df.head())

# Image Prediction
st.subheader("Predict Food Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file and dense_model and cnn_base:
    try:
        # 1️⃣ Load image and resize
        image = Image.open(uploaded_file).convert('RGB').resize((224,224))

        # 2️⃣ Convert to numpy array
        img_array = np.array(image, dtype=np.float32)

        # 3️⃣ Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # 4️⃣ Preprocess for VGG16
        img_array = preprocess_input(img_array)

        # 5️⃣ Extract features using CNN base
        features = cnn_base.predict(img_array)          # shape: (1,7,7,512)
        features_flat = features.reshape(1, -1)         # shape: (1, 25088)

        # 6️⃣ Predict using Dense model
        prediction = dense_model.predict(features_flat)
        predicted_label = label_
