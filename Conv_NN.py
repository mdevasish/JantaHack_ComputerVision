# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 19:51:24 2020

@author: mdevasish
"""


import pandas as pd
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import json

print('Tensorflow script running on version',tf.__version__)

# Extract file names and the labels from the csv file
def extract_images_labels(filename):
    '''Read the image names in the train and test files'''
    print('Present WD is : ',os.getcwd())
    if filename == 'train.csv':
        df = pd.read_csv('./train/'+filename)
        train_images = list(df['image_names'])
        labels = list(df['emergency_or_not']) 
        return train_images,labels
    elif filename == 'test.csv':
        df = pd.read_csv('./train/'+filename)
        test_images = list(df['image_names'])
        return test_images

# Read train image names and their labels & test image names
train_images, labels = extract_images_labels('train.csv')
print('Length of train images',len(train_images))
print('Length of labels:',len(labels))

test_images = extract_images_labels('test.csv')
print('length of test images',len(test_images))

images_path = './train/images'
def read_images(path,image_list):
    '''Read the image files and create a list of images for train and test'''
    images_list = []
    os.chdir(path)
    print('Path changed to : ',os.getcwd())
    for each in image_list:
        images_list.append(plt.imread(each))
    os.chdir('..')
    os.chdir('..')
    print('Path changed to : ',os.getcwd())
    return images_list
    
# Read training images
train_list = read_images(images_path,train_images)

# Read test images
test_list = read_images(images_path,test_images)

print('#'*10+'File read complete''#'*10)

# Building Model Architecture and compiling
model = Sequential()
model.add(Conv2D(128,(4,4),padding = 'same',input_shape = (224,224,3),\
          activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(4,4),padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,(4,4),padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(16,(4,4),padding = 'same',activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(64,activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dropout(0.2))
model.add(Dense(64,activation = 'relu',kernel_initializer = 'uniform'))
model.add(Dropout(0.2))

model.add(Dense(1,activation = 'sigmoid',kernel_initializer = 'uniform'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

print(model.summary())
monitor=EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

hist = model.fit(np.array(train_list),np.array(labels),validation_split = 0.25,\
                 epochs = 10,callbacks = [monitor])
    
def plot_graphs(history, string):
    '''Plot the accuracies to visulaise the underfitting/overfitting issues'''
    plt.subplots(figsize=(24, 12))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history,'accuracy')

predictions = model.predict(np.array(test_list))
#%%
predictions = predictions.reshape(len(predictions),)
fpred = []
for each in predictions:
    if each >= 0.5:
        fpred.append(1)
    else:
        fpred.append(0)

#%%

def submit_results(read_file,write_file):
    '''Store the model architecture into json like structure and write the results'''
    with open('model.json','w') as file:
        file.write(model.to_json())

    model.save_weights('model_weights.h5')
    sub = pd.read_csv(file)

    sub['emergency_or_not'] = fpred

    sub.to_csv(write_file,index = False)

submit_results('sample_submission.csv','Conv_NN_customised.csv')
