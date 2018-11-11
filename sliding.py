import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


from sklearn import svm
import pickle 
#from sklearn.cross_validation import train_test_split


with open('SVM.pkl', 'rb') as f:
    
    clf = pickle.load(f)
    #print(f)
    #clf = pickle.load(f)



def image_resize(image, size=(128, 128)):	
    return cv2.resize(image, size) #, interpolation = cv2.INTER_AREA)




def pyramids(image, scale=1.5, minSize=(128, 128)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image,(w,w))
        if image.shape[0] < minSize[0] or image.shape < minSize[1]:
            break
        yield image






def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])








(winW, winH) = (128, 128)

image = cv2.imread('mug.124.png',0)


small_size = (128, 128)
block_size = (small_size[0] / 2, small_size[1] / 2)
block_stride = (small_size[0] / 4, small_size[1] / 4)
cell_size = block_stride
num_bins = 9
hog = cv2.HOGDescriptor(small_size, block_size, block_stride, cell_size, num_bins)


ins=0
for resized in pyramids(image, scale=1.5):  
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):        
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
		
        clone = resized.copy()
        cv2.rectangle(clone, (x,y), (x+winH, y+winW),(0,255,0),2)        

		
	        
        pix=image_resize(clone[y:y+winH, x:x+winW])
	
        hist = hog.compute(pix)
                
	        
        dataset_size=len(hist)
        rawImages = hist.reshape(dataset_size,-1)
        
        if clf.predict(hist.transpose())=='mug' and clf.predict_proba(hist.transpose())[0][0]>0.50:
           print (clf.predict_proba(hist.transpose()))
           
           #plt.figure(figsize=(15,10))
           #plt.imshow(hist)
         
           #cv2.imwrite("frame"+str(ins)+".jpg", clone)
           #ins+=1
           #cv2.rectangle(resized, (x,y), (x+winH, y+winW),(255,255,0),2)  
           #cv2.imshow("test", clone[y:y+winH, x:x+winW])        
           #cv2.waitKey(0)  
           cv2.imshow("test",clone)
           cv2.waitKey(1)
           time.sleep(0.025)
               

        #cv2.imshow("test",clone)
        #cv2.waitKey(1)
	#time.sleep(0.025)
	
	
            
    
cv2.destroyAllWindows()

