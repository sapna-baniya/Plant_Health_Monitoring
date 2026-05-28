import os
import sys
import tempfile
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath("src"))
from predict_maskrcnn import predict

st.set_page_config(page_title="Plant Health Monitoring", layout="wide")

st.title("Plant Health Monitoring using Deep Learning")
st.write("Upload a plant image and run Mask R-CNN segmentation.")

weights_path = st.text_input(
    "Model weights path",
    "outputs/checkpoints/maskrcnn_best.pth"
)

threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

uploaded = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run Segmentation"):
        if not os.path.exists(weights_path):
            st.error("Weights file not found. Train the model first or provide a valid path.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                input_path = tmp.name

            output_path = "outputs/predictions/streamlit_prediction.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            predict(input_path, weights_path, output_path, score_threshold=threshold)

            st.subheader("Segmentation Output")
            st.image(output_path, use_container_width=True)

            st.subheader("Health Summary")
            st.write("The model highlights healthy leaves, stems, disease spots, stem lesions, and wilting/dead regions.")
