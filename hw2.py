#!/usr/bin/python3

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'notMNIST_sanit.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set tensor',train_dataset.shape,train_labels.shape)
    print('Valid set tensor',valid_dataset.shape,valid_labels.shape)
    print('Test set tensor',test_dataset.shape,test_labels.shape)

    
image_size = 28
num_labels = 10

def reformat(dataset,labels):
    dataset = dataset.reshape((-1,784)).astype(np.float32)
    labels = ( np.arange(num_labels) == labels[:,None] ).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset,train_labels)
valid_dataset, valid_labels = reformat(valid_dataset,valid_labels)
test_dataset, test_labels = reformat(test_dataset,test_labels)

print('Training set tensor',train_dataset.shape,train_labels.shape)
print('Valid set tensor',valid_dataset.shape,valid_labels.shape)
print('Test set tensor',test_dataset.shape,test_labels.shape)

