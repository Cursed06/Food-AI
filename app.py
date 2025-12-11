import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (128, 128) 

st.title("🍱 Food Image Recognition AI")
st.write("Upload a food image and let the AI predict the category.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    class_id = np.argmax(score)
    confidence = float(score[class_id])

    st.subheader("🔍 Prediction")
    st.write(f"**Label:** {class_names[class_id]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    
    st.subheader("📊 Confidence Breakdown")
    for i, c in enumerate(class_names):
        st.write(f"{c}: {float(score[i]):.3f}")
