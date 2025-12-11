import streamlit as st
from PIL import Image
from predict import load_model, load_labels, predict_img

st.set_page_config(page_title="Food Recognition AI", layout="centered")

st.title("🍽️ Food Image Recognition AI")
st.write("Upload an image of food and the AI will classify it.")

model = load_model()
labels = load_labels()

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        preds = predict_img(model, image)

        idx = preds.argmax()
        confidence = preds[idx]

        st.subheader(f"Result: **{labels[idx]}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        # Show top 3 predictions
        st.write("### Top Predictions:")
        top3 = preds.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"{labels[i]} — {preds[i]*100:.2f}%")
