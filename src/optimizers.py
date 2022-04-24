from model import FakeModel
from loss import LossFn
from layers import LinearLayer


class SGD:
  """
    Stochastic Gradient Descent optimizer
    Roll the ball down the hill
  """
  def __init__(self, net: FakeModel, learning_rate=1e-2):
    self.net = net
    self.learning_rate = learning_rate

  def step(self, loss_fn: LossFn):
    gradient = loss_fn.backward()
    for layer in self.net.layers[::-1]:
      gradient = layer.backward(gradient)
      if not isinstance(layer, LinearLayer):
        continue
      layer.apply_delta(self.learning_rate)
