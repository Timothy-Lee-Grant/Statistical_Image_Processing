import numpy as np
import scipy.io as sio
import math

def main():
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']

  #random weight vector
  weightVector = np.ones((10, len(train_img[0])))

  predict(train_img[1], weightVector)

  print("y")

def predict(dataPoint, weightVector):
  dimensions = len(dataPoint)
  prediction = []

  #find total exponent
  totalExp = 0
  for i in range(10):
    for j in range(dimensions):
      totalExp += weightVector[i][j] * dataPoint[j]

  for i in range(10):
    exponent = 0
    for j in range(dimensions):
      exponent += weightVector[i][j] * dataPoint[j]

    value = (math.exp(exponent))/(math.exp(totalExp))
    prediction.append(value)

  print(prediction)
  return(prediction)



if __name__=="__main__":
  main()