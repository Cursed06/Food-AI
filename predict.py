import os
import gdown
import tensorflow as tf
import numpy as np

MODEL_URL = "https://drive.google.com/uc?export=download&id=1uSPfQFZUqbPJyvAjKLPkw0hKfD56XH1q"
MODEL_PATH = "food_model_new.h5"
LABELS_PATH = "labels.txt"

# Auto-download model if missing
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download complete!")

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load labels
def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Predict function
def predict_img(model, img):
    img = img.resize((128, 128))  # match your training size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)[0]
    return preds
