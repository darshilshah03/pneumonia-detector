import numpy as np
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt

from load_dataset import *

train_set_x,train_set_y = train_data()
test_set_x,test_set_y = test_data()

print("Train set X shape : ",train_set_x.shape)
print("Train set Y shape : ",train_set_y.shape)
print("Test set X shape : ",test_set_x.shape)
print("Test set Y shape : ",test_set_y.shape)


model = K.Sequential()

# LAYER 1
model.add(K.layers.Conv2D(filters = 64, kernel_size = (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer=K.initializers.glorot_uniform(),input_shape=(64,64,3)))

model.add(K.layers.MaxPool2D(pool_size = (2,2)))

#LAYER 2
model.add(K.layers.Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding='same' , activation='relu', kernel_initializer=K.initializers.glorot_uniform()))

model.add(K.layers.MaxPool2D(pool_size = (2,2)))



#LAYER 3
model.add(K.layers.Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding='same' , activation='relu', kernel_initializer=K.initializers.glorot_uniform()))


model.add(K.layers.MaxPool2D(pool_size = (2,2)))

#LAYER 4
model.add(K.layers.Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding='same' , activation='relu', kernel_initializer=K.initializers.glorot_uniform()))

model.add(K.layers.MaxPool2D(pool_size = (2,2)))

# #LAYER 5
# model.add(K.layers.Conv2D(filters = 256,kernel_size = (1,1), strides = (1,1), padding='same' , activation='relu', kernel_initializer=K.initializers.glorot_uniform()))
# model.add(K.layers.MaxPool2D(pool_size = (2,2)))

#FLATTEN
model.add(K.layers.Flatten())

model.add(K.layers.Dense(128, kernel_initializer=K.initializers.glorot_uniform(),bias_initializer='zeros',activation='relu'))

model.add(K.layers.Dense(1, kernel_initializer=K.initializers.glorot_uniform(),bias_initializer='zeros',activation='sigmoid'))

datagen = K.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_set_x)

model.compile(optimizer=K.optimizers.Adam(0.001),loss=K.losses.BinaryCrossentropy(),metrics=["accuracy"])

model.fit(datagen.flow(train_set_x,train_set_y,batch_size=32),epochs=20)
print()
result = model.evaluate(test_set_x,test_set_y,verbose=1)

print("Test set loss : ",result[0])
print("Test set accuracy : ",result[1])

print(model.summary())

model.save('./saved_model/model')