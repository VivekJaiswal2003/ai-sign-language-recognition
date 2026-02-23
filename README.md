# AI Sign Language Recognition System

A real-time sign language recognition system that detects hand gestures using MediaPipe hand landmarks and classifies them using machine learning models. The recognized sign is converted into both text and speech output.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Multi-Layer Perceptron (MLP)
- Random Forest Classifier
- StandardScaler
- LabelEncoder
- Text-to-Speech (TTS)

---

##  Project Overview

This system captures live video from a webcam and detects hand landmarks using MediaPipe. The landmark coordinates are converted into numerical features and passed to trained machine learning models for gesture classification.

The predicted sign is displayed as text and converted into speech output.

---

##  Problem Statement

- Communication gap between hearing-impaired individuals and others  
- Lack of accessible real-time interpretation systems  

---

##  System Workflow

1. Webcam captures live video feed  
2. MediaPipe detects hand landmarks  
3. Landmark coordinates are extracted as features  
4. Features are scaled using StandardScaler  
5. Trained ML model predicts the gesture  
6. Output is displayed as text  
7. Text is converted into speech  

---

##  Dataset

- Stored in `asl_landmark_dataset.csv`  
- Contains hand landmark coordinates and corresponding gesture labels  

---

##  Model Details

- Feature scaling and label encoding applied  
- Models trained using:
  - Multi-Layer Perceptron (MLP)
  - Random Forest Classifier  
- Trained model saved and reused for real-time predictions  

---

##  How to Run

1. Clone the repository  
2. Install dependencies  

pip install -r requirements.txt  

3. Run the application  

python main.py  

---

##  Future Improvements

- Increase dataset size  
- Add more gesture classes  
- Deploy as a web-based application  
- Improve model accuracy  



## 👨‍💻 Autho
Vivek Kumar  
B.Tech (ECE) | Machine Learning & DevOps Enthusiast
