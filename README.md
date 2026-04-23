# 🧠 Age & Gender Detection using Deep Learning 

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

**Example Log Entry (CSV):**
Timestamp, Predicted_Age, Gender
2025-04-23 14:03:10, 16, Male
2025-04-23 14:05:15, 12, Female
2025-04-23 14:07:20, 29, Male

---


## ⚠️ Model Bias Notice

The pre-trained Caffe models used in this project exhibit performance bias across different skin tones. They tend to perform more accurately on lighter skin tones and less accurately on darker skin tones, including Indian skin tones.

This limitation arises from the lack of diversity in the training dataset used for these models.

### 🚧 Future Improvements

* Retrain models on more diverse and representative datasets
* Integrate more robust and modern deep learning models
* Apply bias mitigation techniques to improve fairness and accuracy

---

**Note:** This project is intended for educational and demonstration purposes. Predictions may not be reliable across all demographics.



## 👩‍💻 Author

**Aarohi Gaurav Sharma**  
📧 [sharmaaarohigaurav@gmail.com](mailto:sharmaaarohigaurav@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/aarohi-gaurav-sharma-b0a200300)  
💻 [GitHub](https://github.com/aarohi1822)

---

## ⚙️ How to Run the Project

1. Install dependencies:
 `pip install -r requirements.txt `

2. Run the main script:
 `python detect_age_gender.py `

3. Check the `predictions.csv` file for results.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
