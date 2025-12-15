import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi_indonesian_foods.csv"  # CSV with nutritional info

# Google Drive FILE ID for model
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"

IMG_SIZE = (224, 224)  # CNN input size

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_dense_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_cnn_base():
    # CNN feature extractor (VGG16 without top)
    return VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

@st.cache_data
def load_csv():
    # Load nutritional info CSV
    return pd.read_csv(TKPI_PATH, on_bad_lines="skip")

dense_model = load_dense_model()
cnn_base = load_cnn_base()
class_names = load_labels()
tkpi_df = load_csv()

# =========================
# STREAMLIT UI
# =========================
st.title("🍱 Food AI with Nutrition Info")
st.write("Upload a food image and the AI will predict its category and show nutritional information.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    try:
        # --- Display image ---
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # --- Preprocess ---
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # --- Extract features ---
        features = cnn_base.predict(img_array)         # shape: (1,7,7,512)
        features_flat = features.reshape(1, -1)        # shape: (1,25088)

        # --- Predict ---
        preds = dense_model.predict(features_flat)
        probs = tf.nn.softmax(preds[0]).numpy()
        class_id = np.argmax(probs)
        confidence = float(probs[class_id])
        predicted_label = class_names[class_id]

        # --- Display prediction ---
        st.subheader("🔍 Prediction")
        st.write(f"**Label:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        st.subheader("📊 Confidence Breakdown")
        for i, c in enumerate(class_names):
            st.write(f"{c}: {probs[i]:.3f}")

        # --- Display nutritional info from CSV ---
        info = tkpi_df[tkpi_df['Food'].str.lower() == predicted_label.lower()]
        if not info.empty:
            st.subheader(f"📋 Nutritional Information for {predicted_label}")
            st.table(info)
        else:
            st.info("Nutritional info not found in CSV.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
