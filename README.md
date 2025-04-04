#Facial Emotion Detection Project
Overview

This project implements real-time facial emotion detection using mediapipe for face detection and a Convolutional Neural Network (CNN) trained with Keras for emotion classification. The system captures video from a webcam, detects faces, analyze and predicts the emotion expressed on each face.


    emotion_detection.py: The main script for real-time emotion detection using the webcam.

    data_processing.py: Script for preprocessing the fer2013.csv dataset.

    train_model.py: Script for training the CNN model.

    Emotion_Detection.h5: Pre-trained model file for emotion detection.

    haarcascade_frontalface_default.xml: XML file containing data to detect frontal faces.

    fer2013.csv: The dataset with the labeled faces.

#Requirements

Before running the project, ensure you have the following installed:

    Python 3.6 or higher. Im using python 3.11 here( recommended )


Install the required Python packages using pip:


pip install numpy pandas scikit-learn tensorflow keras opencv-python

#Setup and Installation

    Download the required files:

        Download all the files in this repository to your local machine.

        Download the haarcascade_frontalface_default.xml file from the OpenCV GitHub repository (or any reliable source, also given in the repository).

        Ensure that haarcascade_frontalface_default.xml is placed in the same directory as emotion_detection.py. Also make sure to download the X_train.npy and X_test.npy and place them in the same directory (download them from here https://drive.google.com/drive/folders/1ctbQxDDYHdkixhOQD8CkMrwwj_Bk4rRG?usp=sharing ) 

        Obtain the fer2013.csv dataset(from kaggle)

    Create a virtual environment (optional but recommended):
Install the required Python libraries:

pip install -r requirements.txt

If you don't have a requirements.txt file yet, create one by following these steps:

    Generate the requirements file


pip freeze > requirements.txt

Data pre-processing and model trining:

Before running emotion_detection.py, run the data-processing and model trining

    
    python data_processing.py
    python train_model.py

Running the Project

    Run the emotion detection script:


python emotion_detection.py

Using the Webcam:

    The script will start your default webcam.

    It will detect faces and display the predicted emotion in real-time.

    Press q to quit the program.
