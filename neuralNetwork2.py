import numpy as np
import scipy.io as sio

def main():
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']

  print("y")

class LinearLayer:
  #This function should build how many neurons are in the layer
  def __init__(self, inputDim):
    pass



if __name__=="__main__":
  main()