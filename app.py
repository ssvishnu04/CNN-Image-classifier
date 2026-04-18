import streamlit as st
from PIL import Image
from model_helper import predict

# Page config
st.set_page_config(page_title="Vehicle Damage Detection")

# Title + description
st.title("Vehicle Damage Detection")
st.caption("Upload a car image to classify damage type using a CNN model")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            prediction, confidence = predict(image)

        # Correct logic
        if confidence < 0.6:
            st.warning("Not a valid car image. Please upload a clear vehicle image.")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.success(f"Prediction: {prediction}")
            st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
