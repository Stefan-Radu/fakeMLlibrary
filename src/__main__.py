import numpy as np

import os
from model import NetworkModel
from layers import LinearLayer
from activations import Sigmoid, Softmax
from optimizers import SGD
from loss import MSE
from dataloader import MnistDataloader


if __name__ == '__main__':

  ### Data
  input_path = 'data/mnist'
  train_path = os.path.join(input_path, 'train.csv')
  test_path = os.path.join(input_path, 'test.csv')

  mnist_dl = MnistDataloader(train_path, test_path)

  train_data = mnist_dl.get_train_generator()
  test_data = mnist_dl.get_test_generator()

  ### Network
  net = NetworkModel([
    LinearLayer(16),
    Sigmoid(),
    LinearLayer(16),
    Sigmoid(),
    LinearLayer(10),
    Sigmoid(),
  ])

  optim = SGD(net)
  loss_fn = MSE()

  ### Training
  net.train(3, train_data, None, optim, loss_fn)
