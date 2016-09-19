#!/usr/bin/python3

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

import random
import hashlib

def disp_samples(data_folders,sample_size):
    for folder in data_folders:
        print(folder)
        image_files = os.listdir(folder)
        image_sample = random.sample(image_files,sample_size)
        for image in image_sample:
            image_file = os.path.join(folder,image)
            print(image_file)
            #image = mpimg.imread(image_file)
            #plt.imshow(image)
            #plt.show()
            
disp_samples(train_folders,2)
#disp_samples(test_folders,10)

image_size = 28    # pixel width and height
pixel_depth = 255.0 # number of levels per pixel

def load_letter(folder,min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files),image_size,image_size),dtype=np.float32)
    image_index = 0
    print(folder)
    for image in os.listdir(folder):
        image_file = os.path.join(folder,image)
        try:
            image_data = ( ndimage.imread(image_file).astype(float32) - pixel_depth / 2 ) / pixel_depth
            if image_data.shape != (image_size,image_size):
                raise Exception('Unexpected image shape : %s' % str(image_data.shape))
            dataset[image_index,:,:] = image_data
            image_index += 1
        except IOError as e:
            print('could not read:' , image_file, ':', e, ' it\'s okay just skipping.. ')
    num_images = image_index
    dataset = dataset[0:num_images,:,:]
    if num_images < min_num_images:
        raise Exception('Main fewer images than expected %d < %d ' % ( num_images,min_num_images))
    print('Full dataset tensor:' , dataset.shape)
    print('Mean : ', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders,min_num_images_per_class,force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already exists - skipping' % set_filename)
        else:
            print('pickling %s..' % set_filename)
            dataset = load_letter(folder,min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('unable to save data to', set_filename, ':', e)

    


        
