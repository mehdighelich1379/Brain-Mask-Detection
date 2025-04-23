import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Set the page layout to wide for better horizontal space
st.set_page_config(layout="wide")
st.title("Brain Tumor Segmentation")

st.image('MRI_background.jpg' , use_container_width=True)
# Load your trained model
@st.cache_resource
def load_trained_model():
    model = load_model("./models/Mri_Asegmentation.h5")  # Replace with your model's filename
    return model

model = load_trained_model()

# File uploader for MRI image
uploaded_file = st.file_uploader("Please upload your MRI image", type=["jpg", "png", "jpeg" , "tiff"])

if uploaded_file is not None:
    # Open and convert the uploaded image to RGB
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Resize and normalize the image as per your model's requirements
    img_resized = cv2.resize(image_np, (256, 256)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict the mask using your model
    with st.spinner("Predicting mask..."):
        pred_mask = model.predict(img_input)[0]
        pred_mask = (pred_mask > 0.2).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (image_np.shape[1], image_np.shape[0]))
        mask_bw = (pred_mask * 255).astype(np.uint8)

    # Create three columns with specified width ratios
    col1, col2, col3 = st.columns([1, 0.3, 1])

    with col1:
        st.markdown("### Original MRI Image")
        st.image(image, use_container_width=True)

    with col2:
        # Display a centered status message
        st.markdown(
            "<div style='text-align: center; font-size: 20px;'>Analyzing MRI image...</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("### Predicted Tumor Mask")
        st.image(mask_bw, use_container_width=True)

