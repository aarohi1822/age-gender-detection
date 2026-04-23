import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Facial Gesture Recognition")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def load_model():
    return mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def detect(image):
    frame = np.array(image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye = [landmarks[145], landmarks[159]]
            right_eye = [landmarks[374], landmarks[386]]
            mouth = [landmarks[13], landmarks[14]]

            left_eye_ratio = abs(left_eye[0].y - left_eye[1].y)
            right_eye_ratio = abs(right_eye[0].y - right_eye[1].y)
            mouth_open_ratio = abs(mouth[0].y - mouth[1].y)

            if left_eye_ratio < 0.018 and right_eye_ratio < 0.018:
                cv2.putText(frame, "Blinking", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if mouth_open_ratio > 0.05:
                cv2.putText(frame, "Mouth Open", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_container_width=True)

    result = detect(image)
    st.image(result, caption="Processed", use_container_width=True)