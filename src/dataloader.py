import random as rand
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from time import time
from typing import Generator, List, Tuple


#TODO accept function to apply transformatin on training

class MnistDataloader(object):
  def __init__(self, train_path, test_path, batch_size=32, flatten=True, \
               classes=10, train_val_ratio=0.9, randomize=True):
    self.train_path = train_path
    self.test_path = test_path
    self.batch_size = batch_size
    self.flatten = flatten
    self.classes = classes
    self.train_val_ratio = train_val_ratio
    self.randomize = randomize

    self.testing_data = self.read_images_labels(self.test_path, testing=True)
    self.training_data = self.read_images_labels(self.train_path)

  def read_images_labels(self, set_path, testing=False):
    images = []
    labels = []

    with open(set_path, 'r') as f:
      csv_file = csv.reader(f)
      lines = [l for i, l in enumerate(csv_file) if i > 0]

      if self.randomize:
        rand.seed(time())
        rand.shuffle(lines)

      for line in lines:
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

  def get_generator(self, data, batch=False):
    def generator() -> Generator[List, None, None] | \
                       Generator[Tuple, None, None]:
      imgs, labels = data
      img_labels = list(zip(imgs, labels))
      if self.randomize:
        rand.shuffle(img_labels)

      if batch:
        for i in range(0, len(img_labels), self.batch_size):
          end = min(i + self.batch_size, len(imgs))
          yield img_labels[i:end]
      else:
        for img, label in img_labels:
          yield img, label
    return generator()

  def get_train_generator(self):
    images, labels = self.training_data
    l = int(len(images) * self.train_val_ratio)
    images, labels = images[:l], labels[:l]
    return self.get_generator((images, labels), batch=True)

  def get_validation_generator(self):
    images, labels = self.training_data
    l = int(len(images) * self.train_val_ratio)
    images, labels = images[l:], labels[l:]
    return self.get_generator((images, labels))

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
