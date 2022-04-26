import abc
import numpy as np

class LossFn:
  @abc.abstractmethod
  def __call__(self, predicted: np.ndarray, expected: np.ndarray) -> float:
    pass

  @abc.abstractmethod
  def backward(self) -> np.ndarray:
    pass


class MSE(LossFn):
  """
    Mean Squared Error loss function
  """
  def __call__(self, predicted: np.ndarray, expected: np.ndarray) -> float:
    self.error = np.subtract(predicted, expected)
    return np.sum(self.error ** 2)

  def backward(self) -> np.ndarray:
    return 2 * self.error

# TODO more cross entropy loss

if __name__ == '__main__':
  mse = MSE()
  print(mse(np.array([1, 5, 3]), np.array([2, 3, 4])))
  print(mse.backward())


class CrossEntropy(LossFn):
  """
    CrossEntropy Loss function
  """
  def __call__(self, predicted: np.ndarray, expected: np.ndarray) -> float:
    # add 1e-12 to prevent division by 0
    self.error = np.subtract(predicted, expected)
    return -np.mean(expected * np.log(predicted + 1e-12))

  def backward(self) -> np.ndarray:
    return self.error


