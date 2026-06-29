# 😷 Face Mask Detection

A real-time face mask detection system using **Deep Learning and Computer Vision**.

The system detects faces from a camera feed and classifies whether a person is wearing a mask or not using a trained **MobileNetV2 deep learning model**.

## Features

- Real-time face detection using OpenCV
- Mask / No Mask classification
- MobileNetV2 based deep learning model
- Flask web interface
- Detection history logging
- Alert system

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- MobileNetV2

## Installation

Clone the repository:


git clone YOUR_REPO_LINK


Go inside the project folder:


cd Face-Mask-Detection

Install dependencies:


pip install -r requirements.txt

## Run Detection

Run real-time detection:


python detect.py


## Run Flask App

Start the Flask web application:


cd flask_app
python app.py

## Dataset

The dataset is not included because of its large size.

The trained model is already provided, so the project can be run directly without downloading the dataset.

For retraining the model, download a mask and without-mask dataset separately.
