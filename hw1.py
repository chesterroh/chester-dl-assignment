#!/usr/bin/python3

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'http://commondatastorage.googleapis.com/books1000/'

def maybe_download(filename,expected_bytes, force=False):
    if force or not os.path.exists(filename):
        filename, _ = urlretrieve(url+filename,filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and Verified',filename)
    else:
        raise Exception(
            'Failed to verify' + filename + 'please get it with browser')
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)

def maybe_extract(filename,force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        print('%s already present - skipping extraction of %s.' % ( root, filename ))
    else:
        print('Extracing data for %s. This may take a while.' % root )
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root,d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root,d))]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, found %d instead' % (num_classes,len(data_folders)))

    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
