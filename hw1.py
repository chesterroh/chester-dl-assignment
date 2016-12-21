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
            image_data = ( ndimage.imread(image_file).astype(float) - pixel_depth / 2 ) / pixel_depth
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
    return dataset_names

train_datasets = maybe_pickle(train_folders,45000)
test_datasets = maybe_pickle(test_folders,1800)

def disp_number_images(data_folders):
    for folder in data_folders:
        pickle_filename = ''.join(folder) + '.pickle'
        try:
            with open(pickle_filename,'rb') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print('Unable to read data from', pickle_filename, ':', e)
            return
        print('Number of images in ', folder, ':', len(dataset))

disp_number_images(train_folders)
disp_number_images(test_folders)


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows,img_size,img_size), dtype=np.float)
        labels = np.ndarray(nb_rows,dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size = 0 ):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size,image_size)
    train_dataset, train_labels = make_arrays(train_size,image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0 , 0
    end_v, end_t  = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file,'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class,:,:]
                    valid_dataset[start_v:end_v,:,:] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l,:,:]
                train_dataset[start_t:end_t,:,:] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
                
        except Exception as e:
            print('Unable to process data from', pickle_file, ':',e)
            raise

    return valid_dataset,valid_labels,train_dataset,train_labels

train_size = 300000
valid_size = 10000
test_size = 10000

print(train_datasets)
print(test_datasets)

valid_dataset,valid_labels, train_dataset, train_labels = merge_datasets(train_datasets,train_size,valid_size)
_,_,test_dataset,test_labels = merge_datasets(test_datasets,test_size)

print('Training: ', train_dataset.shape,train_labels.shape)
print('Validation: ', valid_dataset.shape,valid_labels.shape)
print('Testing: ', test_dataset.shape,test_labels.shape)





            
