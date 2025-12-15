@@ -1,32 +1,78 @@
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from predict import load_model, load_labels, predict_img
import os
import gdown

st.set_page_config(page_title="Food Recognition AI", layout="centered")
# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"

st.title("🍽️ Food Image Recognition AI")
st.write("Upload an image of food and the AI will classify it.")
# 🔗 Google Drive FILE ID (NOT the full URL)
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
labels = load_labels()

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
# =========================
# LOAD LABELS
# =========================
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # =========================
    # PREPROCESS
    # =========================
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        preds = predict_img(model, image)
    # =========================
    # PREDICT
    # =========================
    preds = model.predict(img_array)
    probs = tf.nn.softmax(preds[0]).numpy()

        idx = preds.argmax()
        confidence = preds[idx]
    class_id = np.argmax(probs)
    confidence = float(probs[class_id])

        st.subheader(f"Result: **{labels[idx]}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔍 Prediction")
    st.write(f"**Label:** {class_names[class_id]}")
    st.write(f"**Confidence:** {confidence:.2f}")

        # Show top 3 predictions
        st.write("### Top Predictions:")
        top3 = preds.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"{labels[i]} — {preds[i]*100:.2f}%")
    st.subheader("📊 Confidence Breakdown")
    for i, c in enumerate(class_names):
        st.write(f"{c}: {probs[i]:.3f}")
