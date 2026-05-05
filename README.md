# 😷 Face Mask Detection using CNN

A deep learning-based web application that detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN).

---

## 📌 Project Overview

This project uses a CNN model trained on image data to classify faces into two categories:
- **With Mask 😷**
- **Without Mask 😐**

The trained model is deployed as a web application where users can upload an image and get real-time predictions.

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV (optional for real-time detection)  
- Streamlit (for deployment)  

---

## 📂 Dataset Structure
dataset/
│
├── with_mask/
│ ├── img1.jpg
│ ├── img2.jpg
│
├── without_mask/
│ ├── img1.jpg
│ ├── img2.jpg


---

## ⚙️ Model Architecture

The CNN model consists of:
- Convolutional layers (feature extraction)
- MaxPooling layers (downsampling)
- Flatten layer (convert to vector)
- Dense layers (classification)
- Dropout (reduce overfitting)

---

## 🔄 Project Workflow
Data Collection → Preprocessing → Model Training → Evaluation → Deployment


---

## 📊 Model Performance

- Training Accuracy: ~98–99%  
- Validation Accuracy: ~96–97%  
- Minor overfitting observed after multiple epochs  

