import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
np.random.seed(95)

def main():
  lambdaVal = 0.02
  stepSize = 0.01

  #load training data's input vector
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']
  #load the labels of each of the images (label indicates the value which the image actually is)
  mat_contents = sio.loadmat('MNIST_train_label.mat')
  trainLabel = mat_contents['trainL']


  #At this piont I have two arrays, one is train_img and the other is trainLabel
  #train_img is
  #trainLabel is an array corresponding to the each element is the actual value of the ith train_img
  ##Testing
  #print(mat_contents)
  #print(f"train_img: {train_img} \n {np.shape(train_img)}")
  #print(f"trainLabel: {trainLabel} \n {np.shape(trainLabel)}")
  #return
  ##End Testing

  #train_img (784X60000)
  #tranLabel (60000X1)

  # **Review** What does this 'splitData' do?
  trainData, testData, trainL, testL = splitData(train_img, trainLabel)

  #create nxn matrix of trigger fucntions
  # **Review** what is a trigger function?
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
      # **Review** what is crossEntropyLoss
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
  # **Review** this must be following an equation given in the notes
  probabilityVector = np.transpose(testData) @ weightVector
  print(probabilityVector.shape)

  prediction = np.zeros((len(testData),1))
  for i in range(len(testData)):
    prediction[i] = np.argmax(probabilityVector[i])




  
#Gradient will point in the direction of greatest assent
# Loss Function: L(W,B) = (1/|T|)* sum (for all training examples) (yi - f(xi;W,B))^2
# The Loss Function is taking an average of the entire dataset where each training example is measured from the amount which the actual output is (yi) from the predicted model output given our parameters
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
  

#Cross Entropy is a metric to mesure the quality of the predicted classification's probabilities
#Cross Entropy is a good method to due to it is non-linear, meaning the derivate will take a bigger "step" when the prediction is "more wrong"
def CaculateCrossEntropyLoss(onMatrix, exponentMatrix):
  #calculate cross entropy loss function
  crossEntropyLoss = np.sum(np.sum(onMatrix@(np.log(exponentMatrix))))
  crossEntropyLoss /= (-1*len(onMatrix))

  return crossEntropyLoss




#I forgot what I was doing with this function
#Trigger matrix must have something to do with on the matrix half of the data is repeated?
#Fills matrix of size dataX10 of 0s, then fills the 
#data: (47988, 1)     It is the # of training examples with each element the label of the particular instance
def findTriggerMatrix(data):

  #onMatrix: (# of train images, 10)
  onMatrix = np.zeros((len(data),10))

  #UNotes: Loops through each of the train image #s
  #  Then each itteration we are changing the content of one element in the onMatrix
  #  We are changing the ith (so corresponding to the # of the training example) and the 
  #  onMatrix will be a matrix for each of the training examples, where all of the contents are 0 except for the index which corresponds to the label which the instance actually is
  for i in range(len(data)):
    onMatrix[i][data[i]] = 1
  return onMatrix



#train_img (784X60000)
#tranLabel (60000X1)
def splitData(data, label):
  trainArray = []
  trainLabel = []
  testArray = []
  testLabel = []

  data = np.transpose(data)

  #split data into two categories: Training and Testing
  #where training has 80% of the data and testing has 20%
  for i in range(len(data)):
    RV = np.random.binomial(1,0.8)
    if RV == 1:
      trainArray.append(data[i])
      trainLabel.append(label[i])
    if RV == 0:
      testArray.append(data[i])
      testLabel.append(label[i])

  #print(f"trainArray: {np.shape(trainArray)}")
  #raise SystemExit
  
  trainData = np.array(trainArray)
  trainLabels = np.array(trainLabel)
  testData = np.array(testArray)
  testLabels = np.array(testLabel)

  #why did I transpose these to get back into that strange situation of rows being the pixels and columns being each of the images?
  trainData = np.transpose(trainData)
  testData = np.transpose(testData)
  return trainData, testData, trainLabels, testLabels

if __name__=="__main__":
  main()