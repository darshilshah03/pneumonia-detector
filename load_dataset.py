import numpy as np

import matplotlib.pyplot as plt
import glob
import cv2

mean = 0
variance = 0
def train_data() :
    global mean,variance
    count1 = 1341
    count2 = 3875
    count = count1 + count2

    arr = np.zeros((count,128,128,3))
    train_set_y = np.ones((1,count))
  

    X_data = []
    files = glob.glob ("dataset/chest_xray/train/NORMAL/*")
    dim = (64,64)
    for myFile in files:
        
        image = cv2.imread (myFile)
        
        im = cv2.resize(image,dim)
        # plt.imshow(im)
        # plt.show()
        X_data.append(im)
    
    files = glob.glob ("dataset/chest_xray/train/PNEUMONIA/*")
    
    for myFile in files:
        
        image = cv2.imread (myFile)
        
        im = cv2.resize(image,dim)
        # plt.imshow(im)
        # plt.show()
        X_data.append(im)
    
    for i in range(count2) : 
        train_set_y[0][i+count1] = 0
    
    arr = np.array(X_data)
    
    
   
    mean = np.mean(arr)
    variance = np.var(arr)
    train_set_x = (arr)/255.0
    
    return train_set_x,train_set_y.T

def test_data() :
    
    
    count1  = 234
    count2 = 390
    count = count1+count2
    

    arr = np.zeros((count,128,128,3))
    train_set_y = np.ones((1,count))
  

    X_data = []
    files = glob.glob ("dataset/chest_xray/test/NORMAL/*")
    dim = (64,64)
    for myFile in files:
        
        image = cv2.imread (myFile)
        
        im = cv2.resize(image,dim)
        # plt.imshow(im)
        # plt.show()
        X_data.append(im)
    
    files = glob.glob ("dataset/chest_xray/test/PNEUMONIA/*")
   
    for myFile in files:
        
        image = cv2.imread (myFile)
        
        im = cv2.resize(image,dim)
        # plt.imshow(im)
        # plt.show()
        X_data.append(im)
    
    for i in range(count2) : 
        train_set_y[0][i+count1] = 0
    
    arr = np.array(X_data)
    
    
    
    mean = np.mean(arr)
    variance = np.var(arr)
    train_set_x = (arr)/255.0
    
    return train_set_x,train_set_y.T

