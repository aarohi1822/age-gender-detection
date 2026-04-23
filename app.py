import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

st.title("Face Detection (Working Version)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect_faces(image):
    img = np.array(image)
    locations = face_recognition.face_locations(img)

    draw = ImageDraw.Draw(image)

    for (top, right, bottom, left) in locations:
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

    return image, len(locations)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_container_width=True)

    result, count = detect_faces(image)
    st.image(result, caption=f"Detected Faces: {count}", use_container_width=True)