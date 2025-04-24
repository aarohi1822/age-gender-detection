# 🧠 Age & Gender Detection using Deep Learning (CCTV-Ready)

This project uses a CNN-based deep learning model to detect a person’s **age range** and **gender** from an image.  
✨ Recently enhanced with **CSV logging** and **alerting system** for detecting **minors (age 0–18)** — ideal for **CCTV-based safety & surveillance** applications.

---

## 📌 Highlights

- 🔍 Detects **Age Group** and **Gender** from facial images
- 🧠 Built with **TensorFlow/Keras**
- 🖼️ Preprocessing done via **OpenCV**
- 🔄 **CSV logging** of every prediction
- 🚨 **Alerts** generated when **minors** (aged 0–18) are detected
- 💾 Model trained/tested on [UTKFace dataset](https://susanqq.github.io/UTKFace/)

---

## 🛡️ Real-World Use Case: Surveillance & Safety

Designed for **installation in CCTV systems** across:
- Schools
- Malls
- Railway stations
- Public safety zones

🔔 **System Behavior**:
- Detects and predicts **age** and **gender**
- If **age is between 0 and 18**, it triggers an **alert**
- Saves all predictions in a **CSV log** file including:
  - Timestamp
  - Predicted Age
  - Predicted Gender

> **Example Log Entry (CSV)**:
```csv
Timestamp, Predicted_Age, Gender
2025-04-23 14:03:10, 16, Male
