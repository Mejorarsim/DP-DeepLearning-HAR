# Project README

## Overview

This project involves video processing, machine learning, and differential privacy. The goal is to create models that can accurately classify video actions while incorporating differential privacy to protect data privacy. The project includes data processing, model training, and evaluation, with an emphasis on differential privacy and data augmentation.

## Directory Structure

- `archive.zip`: Contains the original dataset. [Link: https://drive.google.com/drive/folders/1juFNOYBLiUh2UV1y9E54lwX-F3JhlVtg?usp=sharing]
- `kaggle/input/ucf101-videos`: Directory where the dataset is extracted.
- `train.csv`, `test.csv`: CSV files containing metadata for the training and testing videos.

## Setup Instructions

### Prerequisites

Before running the project, ensure you have the following Python libraries installed:

```bash
pip install imageio opencv-python opencv-contrib-python keras tensorflow opencv-python-headless opacus moviepy
```

### Additional Libraries

```bash
pip install shap numba cloudpickle
```

### Extracting the Dataset

```python
from zipfile import ZipFile

with ZipFile('/nfs/primary/Simran/archive.zip', 'r') as zObject:
    zObject.extractall(path='/nfs/primary/Simran/kaggle/input/ucf101-videos')
```

## Environment Setup

This project requires several Python libraries for video processing, machine learning, and differential privacy.

### Required Libraries

- `imageio`: For reading and writing images and videos.
- `opencv-python` and `opencv-contrib-python`: For advanced image and video processing.
- `keras` and `tensorflow`: For building and training neural network models.
- `opencv-python-headless`: OpenCV version without GUI functionality.
- `opacus`: For adding differential privacy to PyTorch models.
- `moviepy`: For video editing tasks.

### Installing the Libraries

```bash
pip install imageio opencv-python opencv-contrib-python keras tensorflow opencv-python-headless opacus moviepy
```

## Project Workflow

### Importing Required Libraries

```python
import os
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2
import psutil
import time
import tensorflow as tf
from PIL import Image
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from moviepy.editor import VideoFileClip
```

### Data Processing

- **Load datasets**: The training and testing video metadata is loaded from CSV files.
- **Extract frames**: Frames are extracted from videos at specified times for visualization and processing.

### Model Configuration

- **Image Size (IMG_SIZE)**: 224x224 pixels.
- **Batch Size (BATCH_SIZE)**: 128.
- **Epochs (EPOCHS)**: 25.
- **Sequence Length (MAX_SEQ_LENGTH)**: 20.
- **Features (NUM_FEATURES)**: 2048.

### Video Processing Utilities

- `crop_center_square(frame)`: Crops the central square region of the frame.
- `augment_frames(frames)`: Applies random augmentations like flips, brightness adjustment, and rotations.

### Model Training

Four models are trained:
1. **Without Differential Privacy (DP)**
2. **With DP**
3. **With Data Augmentation**
4. **With DP and Data Augmentation**

Each model is evaluated based on accuracy, precision, F1 score, and the impact of DP and data augmentation.

### Visualization

- Training and validation metrics are visualized using plots.
- Confusion matrices and ROC curves are generated for each model.

## Running the Experiment

To run the experiment, execute the following:

```python
# Train and evaluate the models
history_no_dp, model_no_dp = run_experiment(train_data, train_labels_encoded, test_data, test_labels_encoded, use_dp=False)
```

## Results

The models are evaluated based on several metrics:

- **Accuracy**: Overall model performance.
- **F1 Score**: Weighted average of precision and recall.
- **Precision**: Model's ability to not label as positive a sample that is negative.
- **ROC Curves**: Performance of classification models at all classification thresholds.

## Conclusion

This project demonstrates the trade-offs between differential privacy and model performance. Data augmentation can enhance model robustness, and combining it with DP can help protect privacy without significantly sacrificing accuracy.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [OpenCV Documentation](https://docs.opencv.org/master/)