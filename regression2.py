import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
np.random.seed(95)

def main():
  lambdaVal = 0
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']
  mat_contents = sio.loadmat('MNIST_train_label.mat')
  trainLabel = mat_contents['trainL']

  trainData, testData, trainL, testL = splitData(train_img, trainLabel)

  #create nxn matrix of trigger fucntions
  onMatrix = findTriggerMatrix(trainL)

  #random weight vector
  weightVector = np.ones((11,len(trainData)))

  predict(train_img[1], weightVector)

  gradient = findGradient(trainData,testData, onMatrix, weightVector)


def TakeStep(gradientMatrix, stepsize, weightVector):
  return weightVector - (stepsize*gradientMatrix)

def findGradient(trainData, testData, onMatrix, weightVector):
  #add 1 to first entry of all columns (ie, first row should be all 1s)
  #trainData = np.insert(trainData,0,1, axis=1)

  #find exponents
  print(weightVector.shape)
  print(trainData.shape)
  exponentMatrix = weightVector@trainData
  print(exponentMatrix.shape)
  exponentMatrix = np.exp(exponentMatrix)

  denominator = sum(exponentMatrix)
  #denominator = np.expand_dims(denominator, axis=1)
  print(denominator.shape)
  probabilityMatrix = exponentMatrix / denominator
  #trainData = np.insert(trainData,0,1, axis=1)

  print("here")
  print(weightVector.shape)
  print(onMatrix.shape)
  print(probabilityMatrix.shape)
  firstStep = onMatrix - np.transpose(probabilityMatrix)

  gradientMatrix = (trainData @ (onMatrix - np.transpose(probabilityMatrix))) / len(trainData)

  return gradientMatrix

def calculateLoss(trainingSet, onMatrix, lambdaVal):
  runningSum = 0
  for i in range(len(trainingSet)):
    for j in range(len(trainingSet[0])):
      runningSum += 0 #(onMatrix@lnProbabilityMatrix) + lambdaVal(WhatIs beta kj?)

  runningSum /= -len(trainingSet)

  return runningSum

def findTriggerMatrix(data):
  onMatrix = np.zeros((len(data),10))
  for i in range(len(data)):
    onMatrix[i][data[i]] = 1
  return onMatrix


def predict(dataPoint, weightVector):
  prediction = 0
  return(prediction)

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