#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:23:15 2019

@author: prashantsingh
"""

import tensorflow as tf
import matplotlib.image as img
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import models
import cv2


"""
Visualize the data, showing one image per class from 101 classes
"""
rows = 17
cols = 6
fig, ax = plt.subplots(rows, cols, figsize=(25,25))
fig.suptitle("Showing one random image from each class", y=1.05, fontsize=24) # Adding  y=1.05, fontsize=24 helped me fix the suptitle overlapping with axes issue
data_dir = "images/"
foods_sorted = sorted(os.listdir(data_dir))
food_id = 0
for i in range(rows):
  for j in range(cols):
    try:
      food_selected = foods_sorted[food_id] 
      food_id += 1
    except:
      break
    if food_selected == '.DS_Store':
        continue
    food_selected_images = os.listdir(os.path.join(data_dir,food_selected)) # returns the list of all files present in each food category
    food_selected_random = np.random.choice(food_selected_images) # picks one food item from the list as choice, takes a list and returns one random item
    img = plt.imread(os.path.join(data_dir,food_selected, food_selected_random))
    ax[i][j].imshow(img)
    ax[i][j].set_title(food_selected, pad = 10)
    
plt.setp(ax, xticks=[],yticks=[])
plt.tight_layout()
            

"""
splitting the data into train and test set
Helper method to split dataset into train and test folders
"""
def prepare_data(filepath, src,dest):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")
  
print('preparing training data...!')
prepare_data('meta/train.txt', 'images','train')

print('preparing test data')
prepare_data('meta/test.txt', 'images', 'test')


print("Total number of samples in train folder")
file_count = sum([len(files) for r, d, files in os.walk('train')])
print('Number of Files using os.walk in train         :', file_count)


print("Total number of samples in test folder")
file_count = sum([len(files) for r, d, files in os.walk('test')])
print('Number of Files using os.walk in test         :', file_count)


"""
We now have train and test data ready
But to experiment and try different architectures, 
working on the whole data with 101 classes takes a lot of time and computation
To proceed with further experiments, I am creating train_min and test_mini, 
limiting the dataset to 3 classes
Since the original problem is multiclass classification which makes key aspects 
of architectural decisions different from that of binary classification, 
choosing 3 classes is a good start instead of 2

"""


"""
Helper method to create train_mini and test_mini data samples
"""
def dataset_mini(food_list, src, dest):
  if os.path.exists(dest):
    rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
  os.makedirs(dest)
  for food_item in food_list :
    print("Copying images into",food_item)
    copytree(os.path.join(src,food_item), os.path.join(dest,food_item))
    
    
"""
picking 3 food items and generating separate data folders for the same
"""
food_list = []
with open('meta/classes.txt') as file:
    for line in file:
        food_list.append(line.strip())


#food_list = ['apple_pie','pizza','omelette']
src_train = 'train'
dest_train = 'train_mini'
src_test = 'test'
dest_test = 'test_mini'

print("Creating train data folder with new classes")
dataset_mini(food_list, src_train, dest_train)


print("Total number of samples in train_mini folder")
file_count = sum([len(files) for r,d, files in os.walk('train_mini')])
print('Number of files in train_mini : ', file_count)


print("Creating train data folder with new classes")
dataset_mini(food_list, src_test, dest_test)

print("Total number of samples in test_mini folder")
file_count = sum([len(files) for r,d, files in os.walk('test_mini')])
print('Number of files in test_mini : ', file_count)


"""
Keras and other Deep Learning libraries provide pretrained models.
These are deep neural networks with efficient architectures(like VGG,Inception,ResNet) 
that are already trained on datasets like ImageNet.
Using these pretrained models, we can use the already learned weights and 
add few layers on top to finetune the model to our new data.
This helps in faster convergance and saves time and computation when compared to 
models trained from scratch.
We currently have a subset of dataset with 3 classes - samosa, pizza and omelette

"""

K.clear_session()
n_classes = 3
img_width, img_height = 299, 299
train_data_dir = 'train_mini'
validation_data_dir = 'test_mini'
nb_train_samples = 2250 #75750
nb_validation_samples = 750 #25250
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(3,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_3class.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=30,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_3class.hdf5')

class_map_3 = train_generator.class_indices
class_map_3


"""
Visualize the accuracy and loss plots
"""
def plot_accuracy(history,title):
    plt.title(title)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()
def plot_loss(history,title):
    plt.title(title)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()


plot_accuracy(history,'FOOD101-Inceptionv3')
plot_loss(history,'FOOD101-Inceptionv3')

"""The plots show that the accuracy of the model increased with epochs and the loss has decreased
Validation accuracy has been on the higher side than training accuracy for many epochs
This could be for several reasons:
We used a pretrained model trained on ImageNet which contains data from a variety of classes
Using dropout can lead to a higher validation accuracy
Predicting classes for new images from internet using the best trained model
"""


"""
Make a list of downloaded images and test the trained model

"""
images = []

#images.append('apple_pie.jpg')
images.append('p.png')
images.append('125.jpg')
images.append('s.png')


"""
making list of static calories and adding into calories.txt file.
"""
d = {}
with open('calories.txt') as file:
    for line in file:
        key, val = line.split()
        d[key] = val
        


"""
Loading the best saved model to make predictions
"""
K.clear_session()
model_best = load_model('best_model_3class.hdf5',compile = False)


"""Setting compile=False and clearing the session leads to faster loading of the saved model
Withouth the above addiitons, model loading was taking more than a minute!
"""

def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title("food : {} \n calories : {} kcal".format(pred_value,d.get(pred_value)))
        plt.show()



"""
predicting the food category along with calories
"""
predict_class(model_best, images, True)