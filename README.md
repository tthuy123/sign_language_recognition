# Sign Language Recognition System

This project implements a deep learning-based sign language recognition system using MediaPipe for pose detection and LSTM neural networks for sequence classification. The system can recognize sign language gestures from video input.

## Features

- Real-time pose detection using MediaPipe Holistic
- Hand landmark tracking for both hands
- Face and pose landmark detection
- LSTM-based deep learning model for sequence classification
- Support for multiple sign language gestures
- Data collection and preprocessing pipeline
- Model training and evaluation framework

## Requirements

- Python 3.x
- OpenCV (cv2)
- MediaPipe
- TensorFlow
- NumPy
- scikit-learn

Install the required packages using:

pip install -r requirements.txt

## Usage

### 1. Data Collection

To collect keypoint data from sign language videos:

python keypoint_collect.py

This script will:
- Load video URLs from the WLASL100 dataset
- Process each video to extract keypoints using MediaPipe
- Save the keypoint data in the TRAIN_DATASET directory

### 2. Model Training

To train the LSTM model:

python model.py

The training process:
- Loads the collected keypoint data
- Splits data into training and testing sets
- Trains an LSTM model with the following architecture:
  - 3 LSTM layers with batch normalization
  - Dense layers for classification
  - Softmax output layer

## Model Architecture

The model uses a deep LSTM architecture:
- Input shape: (50, 1662) - 50 frames with 1662 keypoints per frame
- 3 LSTM layers with batch normalization
- Dense layers for feature extraction
- Softmax output layer for gesture classification

## Performance

The model's performance can be monitored using:
- TensorBoard logs in the 'Logs' directory
- Test accuracy metrics printed during training
- Early stopping to prevent overfitting
