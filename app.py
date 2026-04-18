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
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Show image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run prediction
        with st.spinner("Analyzing image..."):
            prediction, confidence = predict(image)

        # Handle non-car / low confidence
        if prediction == "Not a valid car image":
            st.warning("⚠️ Please upload a valid car image")
        else:
            st.success(f"Prediction: {prediction}")
            st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
