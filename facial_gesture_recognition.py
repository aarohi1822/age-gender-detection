import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Get key points for gesture detection
            left_eye = [landmarks[145], landmarks[159]]  # Upper and lower eyelid
            right_eye = [landmarks[374], landmarks[386]]  # Upper and lower eyelid
            mouth = [landmarks[13], landmarks[14]]  # Upper and lower lip

            # Calculate distances for eye blinks & mouth openness
            left_eye_ratio = abs(left_eye[0].y - left_eye[1].y)
            right_eye_ratio = abs(right_eye[0].y - right_eye[1].y)
            mouth_open_ratio = abs(mouth[0].y - mouth[1].y)

            # Detect Blinking (adjust threshold if needed)
            if left_eye_ratio < 0.018 and right_eye_ratio < 0.018:
                cv2.putText(frame, "Blinking 👀", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Detect Mouth Open (adjust threshold if needed)
            if mouth_open_ratio > 0.05:
                cv2.putText(frame, "Mouth Open 😮", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Facial Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
