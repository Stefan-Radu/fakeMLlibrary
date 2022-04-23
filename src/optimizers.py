from model import NetworkModel
from loss import LossFn
from activations import Activation


class SGD:
  """
    Stochastic Gradient Descent optimizer
    Roll the ball down the hill
  """
  def __init__(self, net: NetworkModel, learning_rate=1e-2):
    self.net = net
    self.learning_rate = learning_rate

  def step(self, loss_fn: LossFn):
    gradient = loss_fn.backward()
    # TODO remove this
    # print('back loss', gradient)
    # import numpy as np
    # if any(np.isnan(gradient)):
    #   exit(0)
    for layer in self.net.layers[::-1]:
      gradient = layer.backward(gradient)

      if isinstance(layer, Activation):
        continue
      layer.apply_delta(self.learning_rate)
