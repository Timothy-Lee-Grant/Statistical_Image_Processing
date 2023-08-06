import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
np.random.seed(95)

def main():
  lambdaVal = 0.02
  stepSize = 0.01
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']
  mat_contents = sio.loadmat('MNIST_train_label.mat')
  trainLabel = mat_contents['trainL']

  trainData, testData, trainL, testL = splitData(train_img, trainLabel)

  #create nxn matrix of trigger fucntions
  onMatrix = findTriggerMatrix(trainL)

  #add bias value of 1 to each of the instances of the training and testing
  trainData = np.insert(trainData, 0, 1, axis=0)
  testData = np.insert(testData, 0, 1, axis=0)

  #initilize weight vector for each of the classes
  weightVector = np.ones((len(trainData),10))

  crossEntropyLossList = []
  x_axis = []
  #itterate over the gradient direction
  for i in range(1000):
    if i % 100 == 0:
      weightVector, crossEntropyLoss = findGradient(trainData, weightVector, onMatrix, stepSize, lambdaVal, 1)
      crossEntropyLossList.append(crossEntropyLoss)
      x_axis.append(i)
      print(crossEntropyLoss)
    else:
      weightVector, crossEntropyLossDummy = findGradient(trainData, weightVector, onMatrix, stepSize, lambdaVal, 0)

  plt.xlabel('Itteration Number')
  plt.ylabel('Loss Value')
  plt.plot(x_axis,crossEntropyLossList)
  plt.show()

  #make prediction
  #create a v
  probabilityVector = np.transpose(testData) @ weightVector
  print(probabilityVector.shape)

  prediction = np.zeros((len(testData),1))
  for i in range(len(testData)):
    prediction[i] = np.argmax(probabilityVector[i])




  

def findGradient(dataSet, weightVector, onMatrix, stepSize, lambdaVal, entropyNeeded = 0):
  
  #first compute the addition between the weights and the features
  #creates a 10X50k matrix where rows are the different classes for each of the training observations (observations are found in the columns)
  exponentMatrix = np.transpose(weightVector) @ dataSet

  #raise the matrix to the e 
  #result will be e^(b0+b1x1+...+bpxp) with the same structure of training examples down the columns and classes are consistent over the rows
  exponentMatrix = np.exp(exponentMatrix)
  
  #denominator consists of the sum of that observations for all of the different classes (1-10)
  denominator = np.sum(exponentMatrix, axis=0)
  #denominator = denominator[:,np.newaxis]

  exponentMatrix /= denominator

  gradientMatrix = dataSet @ (onMatrix - np.transpose(exponentMatrix))
  gradientMatrix /= -len(dataSet)

  #take a step in the gradient direction
  weightVector = weightVector - (stepSize*gradientMatrix) #changed from - to +, but this resulted in nan so changed it back
  
  #determine if cross entropy is needed to be determined for this gradient pass
  crossEntropyLoss = 0
  if entropyNeeded == 1:
    crossEntropyLoss = CaculateCrossEntropyLoss(onMatrix, exponentMatrix)
    #add regularization term (but doesn't give correct results)
    #crossEntropyLoss += (lambdaVal*weightVector)

  return weightVector, crossEntropyLoss
  

def CaculateCrossEntropyLoss(onMatrix, exponentMatrix):
  #calculate cross entropy loss function
  crossEntropyLoss = np.sum(np.sum(onMatrix@(np.log(exponentMatrix))))
  crossEntropyLoss /= (-1*len(onMatrix))

  return crossEntropyLoss





def findTriggerMatrix(data):
  onMatrix = np.zeros((len(data),10))
  for i in range(len(data)):
    onMatrix[i][data[i]] = 1
  return onMatrix




def splitData(data, label):
  trainArray = []
  trainLabel = []
  testArray = []
  testLabel = []

  data = np.transpose(data)

  #initialize random variable
  for i in range(len(data)):
    RV = np.random.binomial(1,0.8)
    if RV == 1:
      trainArray.append(data[i])
      trainLabel.append(label[i])
    if RV == 0:
      testArray.append(data[i])
      testLabel.append(label[i])
  
  trainData = np.array(trainArray)
  trainLabels = np.array(trainLabel)
  testData = np.array(testArray)
  testLabels = np.array(testLabel)


  trainData = np.transpose(trainData)
  testData = np.transpose(testData)
  return trainData, testData, trainLabels, testLabels

if __name__=="__main__":
  main()