# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Part -1 Building The CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Part - 2 Initialiasing the CNN
classifier = Sequential()
 
 #Step 1 - Convolution
 """
 Theano backend: We use input_shape(3, 64, 64)
 Tensorflow backend: input_shape(64, 64, 3)
 """
 classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation= 'relu'))
 
 #Step 2- Pooling
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
 #Step 3- Flattering
 classifier.add(Flatten())
 
 #Step 4- Full Connection Layers
 classifier.add(Dense(output_dim = 128, activation= 'relu' ))
 classifier.add(Dense(output_dim = 1, activation= 'sigmoid' ))
 
 #Compiling The CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'])

# Part 3 - Fitting the CNN to the images with the images augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)




# Improving our CNN model : Two options : By augmenting the convolution or by adding more full connected layers

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation= 'relu'))
 
 #Step 2- Pooling
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
 # Second convolution layer
 classifier.add(Convolution2D(32, 3, 3, activation= 'relu'))
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
 #Step 3- Flattering
 classifier.add(Flatten())
 
 #Step 4- Full Connection Layers
 classifier.add(Dense(output_dim = 128, activation= 'relu' ))
 classifier.add(Dense(output_dim = 1, activation= 'sigmoid' ))
 
 #Compiling The CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'])

# Part 3 - Fitting the CNN to the images with the images augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


# Making Single prediction4
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[][] == 1 :
    prediction =  'dog'
else:
    prediction =  'cat'