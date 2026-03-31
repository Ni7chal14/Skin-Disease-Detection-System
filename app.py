import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("skin_disease_model.h5")

# Class labels (VERY IMPORTANT)
class_names = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevus",
    6: "Vascular Lesion"
}

st.title("🧠 Skin Disease Detection System")

st.write("Upload an image to detect skin disease")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

   # Prediction
    prediction = model.predict(img_resized)
    probs = prediction[0]

    # Get class + confidence
    class_index = np.argmax(probs)
    confidence = probs[class_index]

    # Gap check
    sorted_probs = np.sort(probs)
    gap = sorted_probs[-1] - sorted_probs[-2]

    # 🚫 Reject if uncertain
    if gap < 0.1:
        st.warning("⚠️ Model is uncertain. Please upload a clearer image.")
        st.stop()

    # Show result
    st.subheader("Prediction:")
    st.write(f"**{class_names[class_index]}**")

    st.subheader("Confidence:")
    st.write(f"{confidence:.2f}")

   # Probabilities
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: {prob:.4f}")