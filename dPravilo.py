import numpy as np
from PIL import *
from  math  import *
import cv2
import os
import matplotlib.pyplot as plt


def D(x, w):
    y=np.zeros(24,dtype=np.float32)

    #print len(w),len(w[0]), len(x)
    for i in range(len(w)):
        for j in range(len(w[i])):
            y[i]+=float(w[i][j]*x[j][0])

    return y

def errorE(d, y):
    E=0.0;
    for i in range(len(d)):
        E+=0.5*((d[i]-y[i])**2)
    return  E
class memory:
    def __init__(self,images,types,labels):
        self.label=[]
        self.typeT=""
        self.image=[]
        self.typeT=types
        self.image=images
        self.label=labels
    '''
    def predict(self, y):
        if  self.temp == y:
            print self.typeT
    '''
    def  input_lerning(self):
        return self.image
    def showIm(self):
        print self.label
        print self.typeT
        plt.imshow(np.reshape(self.image,(28, 28)).astype(np.uint8))
        plt.show()
        plt.close('all')
    def d(self):
        return self.label


def f(x):
    #Relu
    y=[]
    for i in range(len(x)):
        if x[i]>0.5:
            y.append(1)
        else:
            y.append(0)
    return y


def Delta(d, y):
    y2=np.zeros(24,dtype=np.float32)

    for i in range(len(d)):
        y2[i]=(y[i]-d[i])

    return y2

def Lerning(w,x,delta,n=0.01):
    wes=np.zeros((24,784))
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j]=float(w[i][j]-(n*delta[i]*x[j][0]))



input_neural=np.zeros(784,dtype=np.float32)

#f(input_neural.dot(weight))

directory = 'C:/test/5'
files = os.listdir(directory)

k=0
for imagePath in files:
    k+=1

out_neural=np.zeros(k)

weight=np.random.randint(3, size=(24,784))
print weight
neural=[]
s=''
number=1

for imagePath in files:

    image = cv2.imread(directory+'/'+imagePath,0)
    text = imagePath.split(os.path.sep)[-1].split("_")[0]

    if text!=s:
        label = np.zeros(k)
        label[number]=1
        number+=1
        s=text


    im=image/255
    rawImages=im.reshape(784,-1)
    neural.append(memory(rawImages,text,label))











#neural[0].showIm()

E=0
for n  in range(5):
    for i in range(24):
        input_neural=neural[i].input_lerning()

        y=f( D(input_neural,weight))
        print n
        print "Error Vectors :" ,errorE(neural[i].d(),y)
        Lerning(weight,input_neural,Delta(neural[i].d(),y))

input_neural=neural[6].input_lerning()
print f( D(input_neural,weight))
print "_________________________"
print neural[6].d()

