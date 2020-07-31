import numpy as np
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt

from load_dataset import *

train_set_x,train_set_y = train_data()
test_set_x,test_set_y = test_data()

model = K.models.load_model('./saved_model/model')

model.evaluate(train_set_x,train_set_y)

model.evaluate(test_set_x,test_set_y)