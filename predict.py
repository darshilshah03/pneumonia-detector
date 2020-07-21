import numpy as np
import tensorflow as tf
import keras as K
import matplotlib.pyplot as plt
import glob
import cv2
import sys

X_data = []
string = './images/' + sys.argv[1]
files = glob.glob (string)
dim = (64,64)
for myFile in files:
    
    image = cv2.imread (myFile)
    
    im = cv2.resize(image,dim)
    # plt.imshow(im)
    # plt.show()
    X_data.append(im)


data = np.array(X_data)

data  = data/255.0

model = K.models.load_model('./saved_model/model')

results = model.predict(data)
print(results[0][0])
sys.stdout.flush()