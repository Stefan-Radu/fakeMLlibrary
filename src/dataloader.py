import numpy as np
import matplotlib.pyplot as plt
import os
import csv


#TODO split training into train & validation
#TODO option to randomize
#TODO accept function to apply transformatin on training
#TODO normalize

class MnistDataloader(object):
  def __init__(self, train_path, test_path, batch_size=32, flatten=True, \
               classes=10):
    self.train_path = train_path
    self.test_path = test_path
    self.batch_size = batch_size
    self.flatten = flatten
    self.classes = classes

    self.testing_data = self.read_images_labels(self.test_path, testing=True)
    self.training_data = self.read_images_labels(self.train_path)

  def read_images_labels(self, set_path, testing=False):
    images = []
    labels = []

    with open(set_path, 'r') as f:
      csv_file = csv.reader(f)
      for i, line in enumerate(csv_file):
        if i == 0: continue
        label = None
        if not testing:
          label = np.array([0] * self.classes)
          label[int(line[0])] = 1
          img = np.array(line[1:], dtype=float)
        else:
          img = np.array(line, dtype=float)

        if not self.flatten:
          dim = int(img.size ** .5)
          img = img.reshape(dim, -1)

        img /= 255
        images.append(img)
        labels.append(label)

    return images, labels

  def get_generator(self, data):
    def generator():
      imgs, labels = data
      for i in range(0, len(imgs), self.batch_size):
        end = min(i + self.batch_size, len(imgs))
        yield list(zip(imgs[i:end], labels[i:end]))
    return generator()

  def get_train_generator(self):
    return self.get_generator(self.training_data)

  def get_test_generator(self):
    return self.get_generator(self.testing_data)


def test_loader(amount):
  input_path = 'data/mnist'
  train_path = os.path.join(input_path, 'train.csv')
  test_path = os.path.join(input_path, 'test.csv')

  mnist_dl = MnistDataloader(train_path, test_path, flatten=False)

  train = mnist_dl.get_train_generator()
  test = mnist_dl.get_test_generator()

  for batch in test:
    index = 1
    for img, label in batch:
      plt.imshow(img, cmap='gray')
      if label:
        plt.title(label, fontsize = 15)
      if index > amount:
        break
      index += 1
      plt.show()
    break

  for batch in train:
    index = 1
    for img, label in batch:
      print(np.min(img), np.max(img))
      plt.imshow(img, cmap='gray')
      if label[0] is not None:
        print(label)
        plt.title(label, fontsize = 15)
      if index > amount:
        break
      index += 1
      plt.show()
    break


if __name__ == '__main__':
  test_loader(2)
