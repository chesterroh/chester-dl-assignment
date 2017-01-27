#!/usr/bin/python3

from mnist import MnistData

pickle_file = 'notMNIST_sanit.pickle'

mnist_data = MnistData(pickle_file,one_hot=True)

