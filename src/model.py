import json
import numpy as np
from time import time

from layers import LinearLayer
from activations import activations_str_to_class


class FakeModel:
  """
    Build your network here by giving a sequence of
    layers and their corresponding sizes
  """

  def __init__(self, layers):
    self.layers = layers

  @classmethod
  def load(cls, file_name: str):
    """
      read the file (must be json) and load
      the previously saved model
    """
    layers = []
    with open(file_name, 'r') as f:
      data = json.loads(f.read())
      for layer in data.get('layers'):
        typ = layer.get('type')
        if typ == 'linear':
          layers.append(LinearLayer.load(layer))
        else:
          act_class = activations_str_to_class[typ]
          layers.append(act_class())

    return cls(layers)


  def feedforward(self, x: np.ndarray) -> np.ndarray:
    """
      self explanatory. forward pass of the network
    """
    output = x
    for layer in self.layers:
      output = layer(output)
    return output


  def train(self, epochs, train_data, validation_data, optimizer, \
            loss_fn, output_file=''):
    """
      iterate each epoch, batch, image, pass the data
      forward and backwards through the network and update
      the weights
    """
    best_loss = 1e13
    start_time = time()
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
      now_time = time()
      print(f'Epoch {epoch + 1:03d} -> loss: {loss:0.4f}; acc: {acc:0.4f} | ', \
            end=' ')

      val_loss, val_acc = self.validation(validation_data, loss_fn)
      print(f'val_loss: {val_loss:0.4f}; val_acc: {val_acc:0.4f} | ', end= ' ')
      print(f'elasped time: {round(now_time - start_time, 2):0.2f}', flush=True)

      if output_file and val_loss < best_loss:
        # if found better loss update the saved model
        best_loss = val_loss
        self.save(output_file)


  def validation(self, val_data, loss_fn):
    # get validation stats (loss and acc)
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


  def test(self, test_data, output_file: str):
    with open(output_file, "w") as f:
      f.write(f'ImageId,Label\n')
      for i, (img, _) in enumerate(test_data(), 1):
        predicted = self.feedforward(img)
        prediction = np.argmax(predicted)
        f.write(f'{i},{prediction}\n')


  def serialize(self):
    return { 'layers': [ l.serialize() for l in self.layers ] }


  def __repr__(self):
    return json.dumps(self.serialize())


  def save(self, file_name: str):
    # dump the network contents (i.e. all layers contents)
    # to the specified file
    with open(file_name, "w") as f:
      f.write(repr(self))
