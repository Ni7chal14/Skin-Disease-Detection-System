import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown

st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    h1 {
        color: #2c3e50;
    }
    h2, h3 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "skin_disease_model.h5"

@st.cache_resource
def load_skin_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1zGC3VGWtFMqXr66gDWlXACWnQZtut5Q8"
        gdown.download(url, MODEL_PATH, quiet=False)

    return load_model(MODEL_PATH, compile=False)

model = load_skin_model()

# ==============================
# GRAD-CAM FUNCTION
# ==============================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index =  tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ==============================
# CLASS LABELS
# ==============================
class_names = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevus",
    6: "Vascular Lesion"
}

# ==============================
# UI
# ==============================
st.markdown("<h1 style='text-align: center;'>🩺 Skin Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #7f8c8d;'>Upload an image in the sidebar to securely detect and analyze skin conditions.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**How to use:**\n"
    "1. Upload a clear, well-lit photo of the skin lesion.\n"
    "2. The model will analyze the image.\n"
    "3. View the prediction, confidence, and attention map (Grad-CAM).\n\n"
    "*Disclaimer: This tool is for educational purposes and should not replace professional medical advice.*"
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Prediction
    with st.spinner("Analyzing Image..."):
        prediction = model.predict(img_resized)
        probs = prediction[0]

    class_index = np.argmax(probs)
    confidence = probs[class_index]
    predicted_class = class_names[class_index]

    # Gap check
    sorted_probs = np.sort(probs)
    gap = sorted_probs[-1] - sorted_probs[-2]

    if gap < 0.1:
        st.warning("⚠️ **Model is uncertain.** The image might be blurry or the condition is ambiguous. Please upload a clearer image.")
        st.stop()

    # Layout for Image and Grad-CAM
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(img_rgb, use_column_width=True, channels="RGB")

    with col2:
        st.subheader("🔥 Model Attention")
        # ==============================
        # 🔥 GRAD-CAM
        # ==============================
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name:
            heatmap = get_gradcam_heatmap(model, img_resized, last_conv_layer_name)

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Revert img to BGR for adding heatmap since OpenCV uses BGR, then convert back to RGB
            superimposed_img = heatmap * 0.4 + img
            superimposed_img_rgb = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            st.image(superimposed_img_rgb, use_column_width=True)
            st.caption("Highlighted region shows where the model focused.")
        else:
            st.info("Grad-CAM is not available for this model architecture.")

    st.markdown("---")

    # ==============================
    # RESULT SECTION
    # ==============================
    st.subheader("🔬 Diagnosis Results")
    
    # Use metrics for a clean look
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.success(f"**Primary Detection:** {predicted_class}")
    with mcol2:
        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")

    # Probability Distribution
    st.markdown("### Class Probabilities")
    for i, prob in enumerate(probs):
        st.markdown(f"**{class_names[i]}** ({prob * 100:.2f}%)")
        st.progress(float(prob))

    with st.expander("Detailed Analysis Data"):
        st.write("Raw Probabilities Matrix:")
        st.json({class_names[i]: float(probs[i]) for i in range(len(probs))})

else:
    # Display a placeholder when no image is uploaded
    st.info("👈 Please upload an image from the sidebar to begin analysis.")
