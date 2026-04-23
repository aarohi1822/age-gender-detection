import streamlit as st
from PIL import Image, ImageDraw

st.title("Simple Image Analyzer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def simple_demo(image):
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Image Uploaded Successfully")
    return image

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_container_width=True)

    result = simple_demo(image)
    st.image(result, caption="Processed", use_container_width=True)