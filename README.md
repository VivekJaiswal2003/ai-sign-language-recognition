#  AI Sign Language Recognition System

An AI-based **Sign Language Recognition System** that recognizes hand gestures in real time using **Computer Vision and Machine Learning techniques** and converts them into **text and speech output**.

## Overview
Sign language is a primary mode of communication for hearing-impaired individuals, but it is not widely understood by the general public.

This project uses **hand landmark detection** and **feature-based machine learning models** to recognize sign language gestures from a live webcam feed and convert them into readable text and audible speech.

---

## Problem Statement
- Communication gap between hearing-impaired individuals and others  
- Limited availability of real-time sign language interpretation systems  
- Need for an efficient and accessible solution  

---

## Objectives
- Capture hand gestures using a webcam  
- Extract hand landmark features  
- Train machine learning models for gesture classification  
- Display recognized sign as text  
- Convert text into speech using TTS  

---

## Technologies Used
- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- Multi-Layer Perceptron (MLP)  
- Random Forest Classifier  
- StandardScaler  
- Label Encoder  
- Text-to-Speech (TTS)  

---

## System Workflow
1. Webcam captures live video  
2. Hand landmarks are detected and processed  
3. Numerical features are extracted from landmarks  
4. Features are scaled using a feature scaler  
5. Trained model predicts the corresponding sign  
6. Output is displayed as text  
7. Text is converted into speech  

---





## Dataset
- Dataset stored in `asl_landmark_dataset.csv`  
- Contains hand landmark coordinates with corresponding sign labels  

---

## Model Details
- Gesture classification is performed using machine learning models  
- Feature scaling and label encoding are applied before prediction  
- Trained models are saved and reused for real-time detection  

---







