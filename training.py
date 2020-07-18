import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Network import MiniXception
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

"""
before download data using the following URL to the data folder and unpack 'fer2013.csv' file
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules
"""

data_dir = 'data/'
data_filename = 'fer2013.csv'
path = os.path.join(data_dir,data_filename)

logs_path = 'logs/model'

image_size=(48,48)

# load images with faces from csv file 
def load_data():
    data = pd.read_csv(path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).to_numpy()
    return faces, emotions

# scale and normalize input images
def preprocess_input(x):
    x = x.astype('float32')
    x = x / 255.0    
    x = x - 0.5
    x = x * 2.0
    return x

faces, emotions = load_data()
faces = preprocess_input(faces)

# split data on training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

# training parameters and hyperparameters
batch_size = 64
num_epochs = 30
input_shape = (48, 48, 1)
verbose = 1
num_classes = 7
l2_regularization=0.01
patience = 30 # a 'patience' number of epochs
regularization = l2(l2_regularization)

# create data generator for data augmentation
data_generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
                                    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                    zoom_range=.1, horizontal_flip=True)

# build our model calling build method from MiniXception class and compile it
model = MiniXception.build(input_shape[0], input_shape[1], input_shape[2], num_classes, regularization)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# create callbacks
log_file_path = logs_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = logs_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# train the model
History = model.fit_generator(data_generator.flow(xtrain, ytrain,batch_size),
                        steps_per_epoch=len(xtrain) / batch_size, epochs=num_epochs, verbose=1, 
                        callbacks=callbacks, validation_data=(xtest,ytest))

# save weights
model.save_weights('model_weights_2.h5')

# evaluate the model
predictions = model.predict(xtest, batch_size=32)
score = model.evaluate(xtest, ytest,verbose=1)
print('Evaluation score: ' + str(score))

# plot the training loss and accuracy
N = np.arange(0, num_epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, History.history["loss"], label="train_loss")
plt.plot(N, History.history["val_loss"], label="val_loss")
plt.plot(N, History.history["accuracy"], label="train_acc")
plt.plot(N, History.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy (MiniXception)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('Loss_Accuracy_vs_Epoch.jpg')
