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

for imagePath in files:
    
    k+=1
    image = cv2.imread(directory+'/'+imagePath)
   
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    pixels = image_resize(hsv)
    
    hist=hog.compute(pixels)

    
    dataset_size = len(hist)
    rawImages = hist.reshape(dataset_size,-1)
    print clf.predict(hist.transpose())
    cv2.putText(image, clf.predict(hist.transpose())[0], (0, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    '''
    hog.setSVMDetector(np.array(hist))                   
    rects, weights= hog.detectMultiScale(image,  1.5, (7,7),(10,10), 1,1)
    x,y,w,h=0
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print x,y,w,h
    
    '''

  
    
    cv2.imshow(clf.predict(hist.transpose())[0]+str(k),image)

    
cv2.waitKey(0) 
cv2.destroyAllWindows() 













