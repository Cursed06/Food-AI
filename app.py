import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import pandas as pd

# =========================
# CONFIG
# =========================
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"
TKPI_PATH = "tkpi_indonesian_foods.csv"
GDRIVE_MODEL_ID = "1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"
IMG_SIZE = (128, 128)  # model input size

# =========================
# LOAD MODEL
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
@st.cache_data
def load_labels():
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            with open(LABELS_PATH, "r", encoding=enc) as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to load labels: {LABELS_PATH}")
    return []

class_names = load_labels()

# =========================
# LOAD CSV
# =========================
@st.cache_data
def load_csv():
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(TKPI_PATH, encoding=enc, sep=';', on_bad_lines="skip")
            return df
        except UnicodeDecodeError:
            continue
    st.error(f"Failed to load CSV: {TKPI_PATH}")
    return pd.DataFrame()

tkpi_df = load_csv()

# =========================
# FOOD COLUMN
# =========================
food_col = "label" if "label" in tkpi_df.columns else None
if food_col is None:
    st.warning("⚠️ No column named 'label' found in CSV. Nutritional info will not be displayed.")

# =========================
# RULE-BASED DIET RECOMMENDATION
# =========================
def diet_recommendation(row, diet_type="diabetes"):
    calories = float(row['calories'])
    sugar = float(row['sugar'])
    fat = float(row['fat'])
    food_label = row['label'].lower()
    
    if diet_type == "diabetes":
        if sugar <= 5:
            return "Sesuai"
        elif sugar <= 10:
            return "Netral"
        else:
            return "Tidak Disarankan"
    elif diet_type == "vegetarian":
        non_veg = ["ayam", "bebek", "ikan", "rendang", "sate"]
        if any(x in food_label for x in non_veg):
            return "Tidak Disarankan"
        else:
            return "Sesuai"
    elif diet_type == "low_calorie":
        if calories <= 200:
            return "Sesuai"
        elif calories <= 300:
            return "Netral"
        else:
            return "Tidak Disarankan"
    else:
        return "Netral"

# =========================
# STREAMLIT UI
# =========================
st.title("🍱 Food AI with Nutrition Info & Diet Recommendation")
st.write("Upload a food image and the AI will predict its category, show nutritional information, and recommend diet suitability.")

# --- Select diet type ---
diet_type = st.selectbox("Pilih tipe diet:", ["diabetes", "vegetarian", "low_calorie"])

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # --- Display image ---
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # --- Preprocess for model ---
        img_array = np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- Predict ---
        preds = model.predict(img_array)
        probs = tf.nn.softmax(preds[0]).numpy()
        class_id = np.argmax(probs)
        predicted_label = class_names[class_id] if class_names else "Unknown"
        confidence = float(probs[class_id])

        # --- Display prediction ---
        st.subheader("🔍 Prediction")
        st.write(f"**Label:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # --- Confidence breakdown (top 3 other foods) ---
        st.subheader("📊 Confidence Breakdown (Top 3 Other Foods)")
        prob_list = [(c, float(probs[i])) for i, c in enumerate(class_names) if c != predicted_label]
        prob_list.sort(key=lambda x: x[1], reverse=True)
        top3 = prob_list[:3]
        for c, p in top3:
            st.write(f"{c}: {p:.3f}")

        # --- Display nutritional info + diet recommendation ---
        if food_col:
            info = tkpi_df[tkpi_df[food_col].str.lower() == predicted_label.lower()]
            if not info.empty:
                info = info.copy()
                info['Diet Recommendation'] = info.apply(lambda row: diet_recommendation(row, diet_type), axis=1)
                st.subheader(f"📋 Nutritional Information & Diet Recommendation ({diet_type.title()})")
                st.table(info[['food_name', 'calories', 'carbs', 'protein', 'fat', 'sugar', 'sodium', 'Diet Recommendation']])
            else:
                st.info("Nutritional info not found in CSV.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
