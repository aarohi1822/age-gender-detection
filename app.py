import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.title("Age & Gender Detection")

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models (cached)
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb"),
        os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
    )

    age_net = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "age_net.caffemodel"),
        os.path.join(MODEL_DIR, "age_deploy.prototxt")
    )

    gender_net = cv2.dnn.readNet(
        os.path.join(MODEL_DIR, "gender_net.caffemodel"),
        os.path.join(MODEL_DIR, "gender_deploy.prototxt")
    )

    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models()

age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect(image):
    frame = np.array(image)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (227, 227))

            blob_face = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.768, 114.895), swapRB=True
            )

            # Gender
            gender_net.setInput(blob_face)
            gender = gender_labels[gender_net.forward()[0].argmax()]

            # Age
            age_net.setInput(blob_face)
            age = age_labels[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_container_width=True)

    result = detect(image)
    st.image(result, caption="Prediction", use_container_width=True)