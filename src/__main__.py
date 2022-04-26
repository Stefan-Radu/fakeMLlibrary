import os
from model import FakeModel
from layers import LinearLayer
from optimizers import SGD
from loss import MSE, CrossEntropy
from dataloader import MnistDataloader
import activations as act


if __name__ == '__main__':

  ### Data
  input_path = 'data/mnist'
  train_path = os.path.join(input_path, 'train.csv')
  test_path = os.path.join(input_path, 'test.csv')

  no_epochs = 5

  mnist_dl = MnistDataloader(train_path, test_path, batch_size=16)

  train_data = mnist_dl.get_train_generator
  val_data = mnist_dl.get_validation_generator
  test_data = mnist_dl.get_test_generator

  ### Network
  net = FakeModel([
    LinearLayer(100),
    act.ReLU(),
    LinearLayer(10),
    act.Sigmoid()
  ])

  # net = FakeModel.load('./netnet.json')
  # net = FakeModel.load('./netnet27.json') # -> 974s | L 0.02 | Ac 0.9464
  # net = FakeModel.load('./netnet100.json') # -> 412s | 50 epoci | Ac 0.93639 - test

  optim = SGD(net, learning_rate=1e-1)
  loss_fn = MSE()
  # loss_fn = CrossEntropy()

  ### Training
  net.train(no_epochs, train_data, val_data, optim, \
            loss_fn, output_file='./netnet.json')

  # net.test(test_data, './submission.csv')
