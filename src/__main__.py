import numpy as np

import os
from model import FakeModel
from layers import LinearLayer
from optimizers import SGD
from loss import MSE
from dataloader import MnistDataloader
import activations as act


if __name__ == '__main__':

  ### Data
  input_path = 'data/mnist'
  train_path = os.path.join(input_path, 'train.csv')
  test_path = os.path.join(input_path, 'test.csv')

  no_epochs = 50

  mnist_dl = MnistDataloader(train_path, test_path, batch_size=16)

  train_data = mnist_dl.get_train_generator
  val_data = mnist_dl.get_validation_generator
  test_data = mnist_dl.get_test_generator

  ### Network
  net = FakeModel([
    # LinearLayer(784),
    # act.ReLU(),
    LinearLayer(100),
    act.ReLU(),
    LinearLayer(10),
    act.Sigmoid(),
  ])

  optim = SGD(net, learning_rate=1e-1)
  loss_fn = MSE()

  ### Training
  net.train(no_epochs, train_data, val_data, optim, loss_fn)
