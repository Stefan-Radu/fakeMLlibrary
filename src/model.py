import numpy as np


class FakeModel:

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
      no_batches = 0
      for batch in train_data():
        no_batches += 1
        batch_correct = 0
        batch_loss = 0
        for img, label in batch:
          predicted = self.feedforward(img)
          batch_loss += loss_fn(predicted, label)
          batch_correct += (predicted.argmax() == label.argmax())

        loss += batch_loss / len(batch)
        acc += batch_correct / len(batch)
        optimizer.step(loss_fn)

      loss /= no_batches
      acc /= no_batches
      print(f'Epoch {epoch + 1:03d} -> loss: {loss:0.2f}; acc: {acc:0.2f}; ', \
            end=' ')

      val_loss, val_acc = self.validation(validation_data, loss_fn)
      print(f'val_loss: {val_loss:0.2f}; val_acc: {val_acc:0.2f}', flush=True)


  def validation(self, val_data, loss_fn):
    total = 0
    correct = 0
    loss = 0
    for img, label in val_data():
      total += 1
      predicted = self.feedforward(img)
      loss += loss_fn(predicted, label)
      correct += (np.argmax(predicted) == np.argmax(label))

    loss /= total
    acc = correct / total
    return loss, acc
