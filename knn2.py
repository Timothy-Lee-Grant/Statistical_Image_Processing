import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt

def main():
  mat_contents = sio.loadmat('MNIST_train_image.mat')
  train_img = mat_contents['trainX']
  mat_contents = sio.loadmat('MNIST_train_label.mat')
  trainLabel = mat_contents['trainL']

  trainData, testData, trainL, testL = splitData(train_img, trainLabel)


  testData = np.transpose(testData)
  K_set = [1,5,10,20,50,100,200,500]
  y_hat_test = np.zeros( ( len(testData), len(K_set) ) )
  for i in range(100):
  #for i in range(len(testData)):
    x = testData[i]
    x = x[:,np.newaxis]
    distance = np.sum(np.square(trainData - x), axis=1)

    for k in range(len(K_set)):
      K = K_set[k]
      idx = np.argpartition(distance, K)
      y_hat_test[i][k] = np.mean(trainL[idx[np.arange(K)]])

    if i % 10 == 0:
      print("{}, Validation Data".format(i))

  testError = testL*np.ones((1,len(K_set))) - y_hat_test
  testMSE = np.mean(testError**2, axis=0)

  trainDataTranspose = np.transpose(trainData)

  y_hat_train = np.zeros((len(trainDataTranspose),len(K_set)))
  for i in range(100):
  #for i in range(len(trainData)):
    x = trainDataTranspose[i]
    x = x[:,np.newaxis]
    distance = np.sum(np.square(trainData - x), axis=1)

    for k in range(len(K_set)):
      K = K_set[k]
      idx = np.argpartition(distance, K)
      y_hat_train[i][k] = np.mean(trainL[idx[np.arange(K)]])

    if i % 10 == 0:
      print("{}, Training Data".format(i))

  trainError = trainL*np.ones((1,len(K_set))) - y_hat_train
  trainMSE = np.mean(trainError**2, axis=0)


  plt.xlabel('K Value')
  plt.ylabel('Train Mean Squared Error (MSE)')
  plt.plot(K_set, trainMSE)
  plt.show()

  plt.xlabel('K Value')
  plt.ylabel('Validation Mean Squared Error (MSE)')
  plt.plot(K_set, testMSE)
  plt.show()

  #combined graph
  plt.plot(K_set, trainMSE)
  plt.plot(K_set, testMSE)
  plt.xlabel('K Value')
  plt.ylabel('Mean Squared Error (MSE)')
  plt.show()


  #grab test data (real testing, not the cross validation testing)
  mat_contents = sio.loadmat('MNIST_test_image.mat')
  test_img = mat_contents['testX']
  mat_contents = sio.loadmat('MNIST_test_label.mat')
  testLabel = mat_contents['testL']

  test_img = np.transpose(test_img)
  K_set = [1,5,10,20,50,100,200,500]
  y_hat_testReal = np.zeros( ( len(test_img), len(K_set) ) )
  #for i in range(700):
  for i in range(len(test_img)):
    x = test_img[i]
    x = x[:,np.newaxis]
    distance = np.sum(np.square(trainData - x), axis=1)

    for k in range(len(K_set)):
      K = K_set[k]
      idx = np.argpartition(distance, K)
      y_hat_testReal[i][k] = np.mean(trainL[idx[np.arange(K)]])

    if i % 10 == 0:
      print("{}, test Data".format(i))

  testError = testLabel*np.ones((1,len(K_set))) - y_hat_testReal
  testMSE = np.mean(testError**2, axis=0)

  plt.xlabel('K Value')
  plt.ylabel('Test Mean Squared Error (MSE)')
  plt.plot(K_set, testMSE)
  plt.show()







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