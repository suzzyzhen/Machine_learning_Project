#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt


data = genfromtxt('OnlineNewsPopularity.csv', delimiter=',')

dataFeat=data[1:,2:]

def label(data): #if shares >1400, it is popular 
    for x in range(len(data)) :
        if data[:,-1][x] >= 1400:
            data[:,-1][x] = 1
        else:
            data[:,-1][x] = 0
    return(data)

labelData = label(dataFeat)


#use 75% of data for training and 25% of data for teating 

splitIdx = int(len(labelData)*0.75)

trainData = labelData[:splitIdx, :]
testData = labelData[splitIdx:, :]

# ones= np.count_nonzero(trainData[:,-1] == 1) 55% of train data belongs to class 1
# ones= np.count_nonzero(testData[:,-1] == 1) 49% of test data belongs to class 1

def normalization(trainData, testData):
    x_train= trainData[:,:-1]
    x_test = testData[:,:-1]
    mean = np.mean(x_train, axis=0) 
    std = np.std(x_train, axis=0)    
    for i in range(len(trainData)-1):
        trainData[:,:-1][i] = (x_train[i]-mean)/std
    for i in range(len(testData)-1):
        testData[:,:-1][i] = (x_test[i]-mean)/std
    return (trainData, testData)

trainData, testData = normalization(trainData, testData)

xTrain = trainData[:,:-1]
yTrain = trainData[:,-1]

xTest = testData[:,:-1]
yTest = np.array(testData[:,-1])


# In[2]:


def normalization(trainData, testData):
    x_train= trainData[:,:-1]
    x_test = testData[:,:-1]
    mean = np.mean(x_train, axis=0) 
    std = np.std(x_train, axis=0)    
    for i in range(len(trainData)-1):
        trainData[:,:-1][i] = (x_train[i]-mean)/std
    for i in range(len(testData)-1):
        testData[:,:-1][i] = (x_test[i]-mean)/std
    return (trainData, testData)

trainData, testData = normalization(trainData, testData)

xTrain = trainData[:,:-1]
yTrain = trainData[:,-1]

xTest = testData[:,:-1]
yTest = np.array(testData[:,-1])


*************************************Training*************************************

import time
import numpy as np


# Functions and their derivatives
def sigmoid(z): return (1/(1+np.exp(-z)))
def sigmoid_deriv(z): return sigmoid(z)*(1-sigmoid(z))

def tanh(z): return (np.tanh(z))
def tanh_deriv(z): return (1 - tanh(z)**2)

def softplus(z): return np.log(1+np.exp(z))
def softplus_deriv(z): return sigmoid(z)

start = time.time()

def NN(Inputs, Labels, Epochs, MiniBatch, LearningRate, Architecture):
    
    #Number of layers
    Hidden1, Hidden2, Output = Architecture
     
    np.random.seed(1)
    #Weights from norm distr
    WeightLayer1=np.random.normal(0, .1, size=(len(Inputs[0]),Hidden1))
    WeightLayer2=np.random.normal(0, .1, size=(Hidden1, Hidden2))
    WeightLayerOutput=np.random.normal(0, .1, size=(Hidden2, Output))
    
    b1 = np.random.normal(0, .1, size=(1, Hidden1))
    b2 = np.random.normal(0, .1, size=(1, Hidden2))
    b3 = np.random.normal(0, .1, size=(1, Output))
    
    lossTotal = []
     
    for i in range(Epochs):
        lossEpoch = []
        error=0
     
        for batch in range(int(len(Inputs) / MiniBatch)):
            Input=Inputs[(batch*MiniBatch) : ((batch+1)*MiniBatch)]
            Label=Labels[(batch*MiniBatch) : ((batch+1)*MiniBatch)]
            
            #Forward computation
            Layer1 = sigmoid(np.dot(Input, WeightLayer1)+b1)
            Layer2 = sigmoid(np.dot(Layer1, WeightLayer2)+b2)
            Output = sigmoid(np.dot(Layer2,  WeightLayerOutput)+b3)
            
            #Error: target minus output
            Error = (Label - Output)
            lossEpoch.append(np.sum(Error**2))
            
            #Deltas
            TopLayerDelta     = Error * sigmoid_deriv(Output) * Output
            HiddenLayerDelta2 = np.dot(TopLayerDelta, WeightLayerOutput.T) * sigmoid_deriv(Layer2) * Layer2 
            HiddenLayerDelta1 = np.dot(HiddenLayerDelta2,WeightLayer2.T) * sigmoid_deriv(Layer1) * Layer1
            
            db3 = np.sum(TopLayerDelta, axis=0, keepdims=True)
            db2 = np.sum(HiddenLayerDelta2, axis=0)
            db1 = np.sum(HiddenLayerDelta1, axis=0)
            
            #Update the weights
            WeightLayerOutput += LearningRate * np.dot(Layer2.T, TopLayerDelta) 
            WeightLayer2      += LearningRate * np.dot(Layer1.T, HiddenLayerDelta2) 
            WeightLayer1      += LearningRate * np.dot(Input.T, HiddenLayerDelta1)  
            
            b1 += LearningRate * db1
            b2 += LearningRate * db2
            b3 += LearningRate * db3
            
            error += np.sum(np.abs(TopLayerDelta))
            
        #Compute measn sq error 
        mse = np.average(lossEpoch)
        lossTotal.append(mse)
    
    #Return weights, mse and output: everything just to check
    # return ([WeightLayer3,WeightLayer2,WeightLayer1], [mse])
    return([WeightLayer1, WeightLayer2, WeightLayerOutput], [lossTotal], [lossEpoch], [mse], [error], [b1, b2, b3])


#Reconstruct the lables
ytrain=[]
for i in range(len(yTrain)):
    ytrain.append([yTrain[i]])
ytrain=np.array(ytrain)    
  
Arch=[25,15,1]
model=NN(xTrain, ytrain, 800, 2000, 0.01, Arch)

*************************************Testing*************************************

def forward(Input, Weights, Biases):
    
    WeightLayer1, WeightLayer2, WeightLayer3 = Weights 
    b1, b2, b3 = Biases
    
    r0 = sigmoid(np.dot(Input, WeightLayer1)+b1)
    r1 = sigmoid(np.dot(r0, WeightLayer2)+b2)
    r2 = sigmoid(np.dot(r1,  WeightLayer3)+b3)
    
    return(r2)


def Accuracy (In, Out, Weights, Biases):
    predict=forward(In, Weights,Biases) 
    Pred=np.round(predict,0)
    count=0
    for i in range(len(Pred)):
        if Pred[i]==Out[i]:
            count+=1
    return ((count/len(Pred)*100), Pred)

Acc = Accuracy(xTest, yTest, model[0], model[5])
print(Acc[0], '%')

end=time.time()
print(end-start)


# In[3]:


from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(yTest,Acc[1]))
print(classification_report(yTest,Acc[1],digits=4))


# In[ ]:




