import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import gdown

@@ -10,8 +11,9 @@
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi_indonesian_foods.csv"

# 🔗 Google Drive FILE ID (NOT the full URL)
# Google Drive FILE ID (model)
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"

IMG_SIZE = (128, 128)
@@ -23,7 +25,7 @@
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/file/d/1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q/view?usp=sharing"
        url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

@@ -35,6 +37,15 @@ def load_model():
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
@@ -63,17 +74,34 @@ def load_model():
    preds = model.predict(img_array)
    probs = tf.nn.softmax(preds[0]).numpy()

    class_id = np.argmax(probs)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    predicted_label = class_names[class_id]

    # =========================
    # OUTPUT
    # OUTPUT: AI PREDICTION
    # =========================
    st.subheader("🔍 Prediction")
    st.write(f"**Label:** {class_names[class_id]}")
    st.write(f"**Label:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("📊 Confidence Breakdown")
    for i, c in enumerate(class_names):
        st.write(f"{c}: {probs[i]:.3f}")
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
