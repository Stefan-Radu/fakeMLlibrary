import numpy as np

# import layers as L
# from typing import List


class NetworkModel:
  def __init__(self, layers):
    self.layers = layers

  def feedforward(self, x: np.ndarray) -> np.ndarray:
    output = x
    for layer in self.layers:
      output = layer(output)
    return output

  def train(self, epochs, train_data, validation_data, optimizer, loss_fn):
    for epoch in range(epochs):
      loss, acc = 0, 0
      for batch in train_data:
        batch_correct = 0
        batch_loss = 0
        for img, label in batch:
          # print(img.shape)
          # print(label.shape)
          predicted = self.feedforward(img)
          loss += loss_fn(predicted, label)
          batch_correct += (predicted.argmax() == label)

        loss += batch_loss / len(batch)
        acc += batch_correct / len(batch)
        optimizer.step(loss_fn)

      print(f'Epoch {epoch + 1} -> loss: {loss}; acc: {acc}', flush=True)

  # def test_acc(net: nn.Module, test_loader: DataLoader):
  #   net.eval()
  #
  #   total = 0
  #   correct = 0
  #
  #   for test_images, test_labels in test_loader:
  #     total += len(test_images)
  #     out_class = torch.argmax(net(test_images))
  #     correct += torch.sum(out_class == test_labels)
  #
  #   return correct / total * 100
