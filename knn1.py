import numpy as np
import scipy.io as sio
import math

def main():
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']
  mat_contents = sio.loadmat('MNIST_train_label.mat')
  trainLabel = mat_contents['trainL']

  trainData, testData, trainL, testL = splitData(train_img, trainLabel)

  testData = np.transpose(testData)
  distance = findDistance(trainData, np.transpose(trainData[9]))
  print(distance.shape)
  #raise Exception("stop")
  #calculate distance array, 

  #once I find distance I don't need to keep the other ones, all I need to do is make the decision for that one test point, and then I can use that to find the the different for the different k values

  distance = []
  for i in range(len(testData)):
    distance.append(findDistance(trainData, testData[i]))
    if i % 10 == 0:
      print(i)

  #test each of the k values
  for k in [1,5,10,20,50,100,200,500]:
    predict(distance, k, trainL)




def findDistance(dataSet, point):

  point = point[:,np.newaxis]
  #print(point.shape)
  #print(dataSet.shape)
  difference = np.power(np.subtract(point,dataSet),2)
  #print("before")
  #print(difference.shape)
  distance = np.sum(difference, axis=0 )
  print(np.argsort(distance)) #this can not be true because the shortest distance should always be to itself (so the index which it is associated with)

  return distance
  
def predict(distance, k, trainL):
  return 0


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