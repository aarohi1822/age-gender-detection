import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.title("Age & Gender Detection")

# === Paths ===
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "models")

# === Load Models ===
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(
        os.path.join(model_path, "opencv_face_detector_uint8.pb"),
        os.path.join(model_path, "opencv_face_detector.pbtxt")
    )

    age_net = cv2.dnn.readNet(
        os.path.join(model_path, "age_net.caffemodel"),
        os.path.join(model_path, "age_deploy.prototxt")
    )

    gender_net = cv2.dnn.readNet(
        os.path.join(model_path, "gender_net.caffemodel"),
        os.path.join(model_path, "gender_deploy.prototxt")
    )

    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models()

# === Labels ===
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# === Upload Image ===
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect(image):
    frame = np.array(image)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    male_count, female_count = 0, 0
    minor_detected = False

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
            gender_preds = gender_net.forward()
            gender = gender_labels[gender_preds[0].argmax()]

            # Age
            age_net.setInput(blob_face)
            age_preds = age_net.forward()
            age = age_labels[age_preds[0].argmax()]

            # Count
            if gender == "Male":
                male_count += 1
            else:
                female_count += 1

            if age in ['(0-2)', '(4-6)', '(8-12)']:
                minor_detected = True

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, male_count, female_count, minor_detected

# === Run Detection ===
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    result_img, males, females, minor = detect(image)

    st.image(result_img, caption="Processed Image", use_container_width=True)
    st.write(f"Males: {males}")
    st.write(f"Females: {females}")

    if minor:
        st.error("⚠️ Minor Detected")