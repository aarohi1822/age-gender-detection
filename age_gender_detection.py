import cv2
import numpy as np
import os
from datetime import datetime

# === Paths ===
base_path = "/Users/aarohigauravsharma/AgeGenderDetection/"  # your local path

# === Load Models ===
face_net = cv2.dnn.readNet(os.path.join(base_path, "opencv_face_detector_uint8.pb"),
                           os.path.join(base_path, "opencv_face_detector.pbtxt"))
age_net = cv2.dnn.readNet(os.path.join(base_path, "age_net.caffemodel"),
                          os.path.join(base_path, "age_deploy.prototxt"))
gender_net = cv2.dnn.readNet(os.path.join(base_path, "gender_net.caffemodel"),
                             os.path.join(base_path, "gender_deploy.prototxt"))

# === Labels ===
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# === Webcam Setup ===
cap = cv2.VideoCapture(1)  # default webcam

def detect_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    return face_net.forward()

def preprocess_face(face):
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        return cv2.resize(face, (227, 227))
    except:
        return None

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detections = detect_faces(frame)

    # === Reset counters per frame ===
    male_count, female_count = 0, 0
    minor_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            face = frame[y1:y2, x1:x2]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face = preprocess_face(face)
            if face is None:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         (78.426, 87.768, 114.895), swapRB=True)

            # Gender prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender_conf = gender_preds[0].max()
            gender = gender_labels[gender_preds[0].argmax()] if gender_conf > 0.6 else "Uncertain"

            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_labels[age_preds[0].argmax()]

            # Count
            if gender == "Male":
                male_count += 1
            elif gender == "Female":
                female_count += 1

            # Minor alert
            if age in ['(0-2)', '(4-6)', '(8-12)']:
                minor_detected = True

            # Label & Box
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Logging (optional)
            with open("log.csv", "a") as log:
                log.write(f"{datetime.now()},{gender},{age}\n")

    # === Display Frame Data ===
    cv2.putText(frame, f"Males: {male_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Females: {female_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if minor_detected:
        alert = "⚠️ ALERT: Minor Detected!"
        cv2.putText(frame, alert, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(alert)
        with open("alerts.csv", "a") as alert_log:
            alert_log.write(f"{datetime.now()},Minor Detected\n")

    # === Show Window ===
    cv2.imshow("Smart City AI - Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
