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

GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"
IMG_SIZE = (224, 224)  # CNN input size

# =========================
# LOAD MODELS & DATA
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
    return VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

@st.cache_data
def load_labels():
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            with open(LABELS_PATH, "r", encoding=enc) as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to load labels: {LABELS_PATH}")
    return []

@st.cache_data
def load_csv():
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:
            return pd.read_csv(TKPI_PATH, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to load CSV: {TKPI_PATH}")
    return pd.DataFrame()

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

        # --- Preprocess image ---
        img_resized = img.resize((128,128))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1,128,128,3)
        
        # --- Predict directly ---
        preds = model.predict(img_array)

        probs = tf.nn.softmax(preds[0]).numpy()
        class_id = np.argmax(probs)
        confidence = float(probs[class_id])
        predicted_label = class_names[class_id] if class_names else "Unknown"

        # --- Show prediction ---
        st.subheader("🔍 Prediction")
        st.write(f"**Label:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # --- Show confidence breakdown ---
        st.subheader("📊 Confidence Breakdown")
        for i, c in enumerate(class_names):
            st.write(f"{c}: {probs[i]:.3f}")

        # --- Show nutritional info ---
        info = tkpi_df[tkpi_df['Food'].str.lower() == predicted_label.lower()]
        if not info.empty:
            st.subheader(f"📋 Nutritional Information for {predicted_label}")
            st.table(info)
        else:
            st.info("Nutritional info not found in CSV.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

