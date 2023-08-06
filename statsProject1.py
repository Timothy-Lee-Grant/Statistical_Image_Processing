import numpy as np
import scipy.io as sio

def main():
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']

  print("y")

class LinearLayer:
  def __init__(self, inputDim):
    pass

def calculateLoss(trainingSet):
  dim = len(trainingSet[0])
  loss = 0
  for i in range(len(trainingSet)):
    for j in range(len(dim)):
      inner = 0

if __name__=="__main__":
  main()