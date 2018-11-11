import os
import cv2
import numpy as np
from sklearn import svm
import pickle
 
from sklearn.cross_validation import train_test_split




def image_resize(image, size=(128, 128)):
    return cv2.resize(image, size)


directory = 'C:/digits' 
files = os.listdir(directory)


small_size = (128, 128)
block_size = (small_size[0] / 2, small_size[1] / 2)
block_stride = (small_size[0] / 4, small_size[1] / 4)
cell_size = block_stride
num_bins = 9
hog = cv2.HOGDescriptor(small_size, block_size, block_stride, cell_size, num_bins)

with open('SVM.pkl', 'rb') as f:
    clf = pickle.load(f)

k=0


cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   
        pixels = image_resize(hsv)
        hist=hog.compute(pixels)   
        dataset_size = len(hist)
        rawImages = hist.reshape(dataset_size,-1)
        #print clf.predict(hist.transpose())
        cv2.putText(frame, str(clf.predict(hist.transpose())[0])+"="+str(clf.predict_proba(hist.transpose())[0][0]), (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(10) ==27:
            break

    

cv2.destroyAllWindows() 













