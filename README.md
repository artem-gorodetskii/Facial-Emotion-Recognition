# Facial-Emotion-Recognition
Real time facial emotion recognition using MiniXception network architecture. 
Face detection was performed using OpenCV Haar Cascade. 
You can test it directly using your webcam in real time.

1. Before training the model please download dataset from Kaggle competiotion
  using the following URl: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
2. Unpack the 'fer2013.tar.gz' file.
3. The 'Network.py' file contains the model class with static build method.
3. The 'training.py' contains python code for training the model (using tensorflow framework).
4. 'haarcascade_frontalface_default.xml' is a cascade for face detection.
4. The 'logs' directory used for model checkpoints.
5. 'model_weights.h5' represents the weights of trained model.
6. 'emotion_recognition_webcam.ipynb' contains code for testing the trained model and allows real time emotion recognition using your webcam device.

