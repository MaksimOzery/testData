import os
import cv2
import numpy as np
from sklearn import svm
import pickle
 
from sklearn.cross_validation import train_test_split




def image_resize(image, size=(150, 150)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size)


directory = 'C:/train' 
files = os.listdir(directory)

labels = []
rawImages = []

small_size = (128, 128)
block_size = (small_size[0] / 2, small_size[1] / 2)
block_stride = (small_size[0] / 4, small_size[1] / 4)
cell_size = block_stride
num_bins = 9
hog = cv2.HOGDescriptor(small_size, block_size, block_stride, cell_size, num_bins)

clf = svm.SVC(kernel='linear',C=1.0, random_state=0,probability=True)

k=0
for imagePath in files:
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    k+=1
    image = cv2.imread(directory+'/'+imagePath,0)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]    
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    
    pixels = image_resize(image)
    
    hist=hog.compute(pixels)
    #k+=hist.nbytes/ (1024 * 1000.0)
   
    rawImages.append(hist)
   
    labels.append(label)
    
    #print imagePath
    if cv2.waitKey(10) == 27:
        cv2.imshow('image',pixels)

rawImages = np.asarray(rawImages)
labels = np.asarray(labels)

dataset_size = len(rawImages)
rawImages = rawImages.reshape(dataset_size,-1)

dataset_size = len(labels)
labels = labels.reshape(dataset_size,-1)


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)


clf.fit(trainRI, trainRL)

with open('SVM.pkl', 'wb') as f:
    pickle.dump(clf, f)
    
   
acc = clf.score(testRI, testRL)    
#y_test=clf.predict( trainRI)    
print("[INFO] accuracy: {:.2f}%".format(acc * 100))
