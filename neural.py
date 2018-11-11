import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn import neural_network
from sklearn.linear_model import Perceptron



from sklearn.cross_validation import train_test_split





directory = 'C:/test/5' 
files = os.listdir(directory)

labels = []
rawImages = []


k=0
neural_network_clf = neural_network.MLPClassifier()



for imagePath in files:

    k+=1
    image = cv2.imread(directory+'/'+imagePath,0)
    label = imagePath.split(os.path.sep)[-1].split("_")[0]    
   
    print label
    
    rawImages.append(image/255)
   
    labels.append(label)
    '''
    #print imagePath
    if cv2.waitKey(0):
        cv2.imshow('image',image)
    '''

rawImage = np.asarray(rawImages)

labels = np.asarray(labels)

dataset_size = len(rawImage)
rawImages = rawImage.reshape(dataset_size,-1)

dataset_size = len(labels)
#labels = labels.reshape(dataset_size,-1)

#раскидать примеры случайно, но из-за малого  количества не имеет смысла
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)

print len(trainRL)
print len(trainRI)

#обучение( trainRL,trainRL)
neural_network_clf.fit(rawImages, labels)

#оценка нейросети( testRI,trainRI)
#score = neural_network_clf.score(rawImages, labels)    

n=5

#предсказание
print("\nTest :", neural_network_clf.predict(rawImages[n])[0])

k=np.reshape(rawImages[n],(28, 28)).astype(np.uint8)
print k
plt.imshow(k)
plt.show()
plt.close('all')
#можно сделать оценку , но не имеет сысла из-за малого  количества примеров
#print('Test accuracy:', score[1])




