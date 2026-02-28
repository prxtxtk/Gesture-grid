# 🤟 ASL Gesture Recognition System

A real-time American Sign Language (ASL) gesture recognition system built using Python, OpenCV, and Machine Learning.

This project captures hand gestures through a webcam and predicts the corresponding ASL alphabet using a trained ML model. It combines computer vision, feature extraction, and a web-based interface for interactive predictions.

---

## 🚀 Features

- 🎥 Real-time webcam gesture capture
- 🧠 Machine Learning-based gesture classification
- 📊 Pre-trained model using `.joblib`
- 🌐 Interactive frontend (HTML, CSS, JavaScript)
- 🔄 Python backend for processing and prediction

---

## 🏗️ Project Structure

ASL-Detection-System/

├── server.py  
├── train.py  
├── cam.py  
├── asl_model_improved_continued.joblib  
├── index.html  
├── style.css  
├── script.js  
├── images/  
├── requirements.txt  
└── README.md  

---

## 🧠 How It Works

1. Webcam captures hand gesture frames.
2. Image preprocessing and feature extraction are applied.
3. The trained ML model predicts the corresponding ASL character.
4. The prediction result is displayed on the web interface.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python server.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🛠️ Technologies Used

- Python
- OpenCV
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript

---

## 📈 Future Improvements

- Improve accuracy using deep learning (CNN)
- Add word prediction from letter sequences
- Improve lighting and background robustness
- Deploy to cloud for public access
- Add mobile compatibility

---

## 🎯 Use Cases

- Assistive communication technology
- Educational tool for learning ASL
- Human-computer interaction research
- Computer vision experimentation

---

## 👨‍💻 Author

Prateek Kumar  
Engineering Student at CMR institute of technology,Bangalore
