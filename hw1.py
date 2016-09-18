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


    
 
