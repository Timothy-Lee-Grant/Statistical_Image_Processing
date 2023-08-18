import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
np.random.seed(95)

stepSize = 0.001

def main():
  lambdaVal = 0.02
  stepSize = 0.01
  numberOfPixels = 784
  numberOfClasses = 10

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

  #At this point I now have the data segmented. That is all I have,
  #I will need to First train my B parameters for each of the pixel values (for each of the classes?)

  betaParameters = np.ones((numberOfPixels + 1, numberOfClasses))

  inputParameters = trainData
  #I want to insert a 1 into the first row
  numberOfTrainingPoints = inputParameters.shape[1]
  onesVector = np.ones((1,numberOfTrainingPoints))
  inputParameters = np.insert(inputParameters, 0, onesVector, axis=0)

  #create trigger matrix
  triggerMatrix = CreateTriggerMatrix(trainL)

  #probability matrix (might need to keep calling this function)
  #probabilityMatrix = FillInProbabilityMatrix(inputParameters[:,1], betaParameters)
  #betaParameters, crossEntropyLoss = PerformGradientDecent(betaParameters, triggerMatrix, inputParameters)


  #Keep itterating through gradient decent until stopping criteia
  #Each itteration keep track of the cross entropy loss to be able to plot over intervals
  x_axis = []
  crossEntropyLossList = []
  for i in range(20):
    betaParameters, crossEntropyLoss = PerformGradientDecent(betaParameters, triggerMatrix, inputParameters)
    x_axis.append(i)
    crossEntropyLossList.append(crossEntropyLoss)
    print(f"{i}: {crossEntropyLoss}")

  #exit()

  plt.xlabel('Itteration Number')
  plt.ylabel('Loss Value')
  plt.plot(x_axis,crossEntropyLossList)
  plt.show()

  exit()

  #make prediction
  # **Review** this must be following an equation given in the notes
  probabilityVector = np.transpose(testData) @ weightVector
  print(probabilityVector.shape)

  prediction = np.zeros((len(testData),1))
  for i in range(len(testData)):
    prediction[i] = np.argmax(probabilityVector[i])


def PerformGradientDecent(betaParameters, triggerMatrix, inputParameters):
  
  numberOfSamples =  inputParameters.shape[1]

  #First fill out the equation for the probability matrix
  #Starting get the exponent, where each exponent is the betaPara multiplies by the pixel
  #At end of this line, 10X50K matrix.
  #  Row of each class, column of each of the training example
  probabilityMatrix = np.transpose(betaParameters) @ inputParameters

  #saving for CrossEntropyLoss
  nonExponential = probabilityMatrix

  #Put each element as an exponent (as specified by the formula)
  probabilityMatrix = np.exp(probabilityMatrix)

  #Now I need to divide by the sum of the classes
  #Nieve way would be to do 50k summations and then divide
  # **CHECK** This is probably incorrect
  #Idea is to sum up each of the columns, and for each element in the probMat divide by that associated sum term
  probabilityMatrix = probabilityMatrix / np.sum(probabilityMatrix, axis=0)

  gradientMatrix = (-1/numberOfSamples) * inputParameters @ (triggerMatrix - np.transpose(probabilityMatrix) )
  
  #I now created the gradient matrix according to the equations in the notes
  #now I need to take those betaParameters (the values used to predicted) and move them in the direction of the gradient
  betaParameters = betaParameters - (stepSize * gradientMatrix)

  print("about to calculate the CEL")

  #calculate cross entropy loss
  #CEL is defined as -(1/n) * sum over all samples * sum over all classes (trigger matrix * ln (probmatrix))
  crossEntropyLoss = triggerMatrix @ nonExponential
  print(f"triggerMatrix.shape: {triggerMatrix.shape}")
  print(f"nonExponential.shape: {nonExponential.shape}")
  #crossEntropyLoss = nonExponential
  print("first")
  crossEntropyLoss = np.sum(crossEntropyLoss, axis=0)
  print("second")
  print(f"crossEntropyLoss.shape: {crossEntropyLoss.shape}")
  crossEntropyLoss = np.sum(crossEntropyLoss)
  print("third")
  crossEntropyLoss = (-1/numberOfSamples) * crossEntropyLoss
  print("fourth")

  return betaParameters, crossEntropyLoss




def FillInProbabilityMatrix(inputParameters, betaParameters):
  numberOfClasses = betaParameters.shape[1]
  print(f"inside of prob; {numberOfClasses}")
  print(betaParameters.shape)
  betaParameters = betaParameters.transpose()
  print(betaParameters.shape)
  print(inputParameters.shape)
  #first find the sum
  i = 1
  sumValue = 0
  for i in range(numberOfClasses):
    sumValue += np.exp( np.sum(betaParameters[i]@inputParameters) )
  #np.sum(betaParameters[i]@inputParameters)

  print(f"here it is {sumValue}")
    

  return None

#This function will take in a vector which has the value of the assocated 
def CreateTriggerMatrix(trainL):
  triggerMatrix = np.zeros(( len(trainL), 10))

  #loop through each of the indicies of the trainL array and create a new row each time
  i = 0
  for i in range(len(trainL)):
    triggerMatrix[i][trainL[i]] = 1
    
  return triggerMatrix


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