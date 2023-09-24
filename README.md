# Emotion_recognition
### Emotion Recognition through Grayscale classification using CNN
## Overview

This project aims to classify human emotions based on facial expressions using a Convolutional Neural Network (CNN). The model takes grayscale facial images as input and classifies them into one of seven categories: Happy, Sad, Angry, Surprise, Fear, Disgust, and Neutral. It utilizes TensorFlow and Keras libraries for model building and training.

## Table of Contents

1. Prerequisites
2. Installation
3. Project Structure
4. Usage
5. Model Architecture
6. Training
7. Results
8. License


## 1. Prerequisites

* Python 3.x
* TensorFlow 2.x
* Keras
* NumPy
* Matplotlib
* seaborn
* livelossplot

## 2. Installation
* Clone this repository:
  ''' git clone https://github.com/shreyas-chigurupati07/emotion-recognition.git

* Navigate to the project directory:
  ''' cd emotion-recognition

* Install the required packages: 
  ''' pip install -r requirements.txt

## 3. Project structure
* 'train/': Contains training images organized by emotion categories.
* 'test/': Contains test images for validation.
* 'model_weights.h5': Pre-trained model weights.
* 'model.json': Model architecture.
* 'main.py': Main Python script to run the code.

## 4. Usage
* To train the model, run:
  ''' python main.py --mode train
* To evaluate the model, run:
  ''' python main.py --mode evaluate

## 5. Model Architecture
The CNN model has four convolutional layers, followed by max-pooling and dropout layers for down-sampling and regularization. The architecture also includes two fully-connected layers for classification. For more details, please refer to the code.


## 6. Training
The model is trained using Adam optimizer with a learning rate of 0.0005 and categorical cross-entropy as the loss function. The training also uses callbacks for learning rate reduction and checkpointing the best model.

## 7.Results
The model achieved an accuracy of 83% on the validation set.

## 8. License
This project is licensed under the MIT License - see the LICENSE.md file for details.


