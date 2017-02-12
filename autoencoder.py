#!/usr/bin/python3

from mnist import MnistData
import tensorflow as tf
import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

K = 1000
L = 500
M = 250

learning_rate = 1e-4
batch_size = 100
total_batch = int(mnist.train_data.data_length/batch_size)
train_iteration = 5

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])

W1_encoder = tf.Variable(tf.truncated_normal([784,K],stddev=0.1))
B1_encoder = tf.Variable(tf.zeros([K]))

W2_encoder = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
B2_encoder = tf.Variable(tf.zeros([L]))

W3_encoder = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
B3_encoder = tf.Variable(tf.zeros([M]))

def encoder( x ):
    H1_encoder = tf.nn.sigmoid(tf.matmul(x,W1_encoder)+B1_encoder)
    H2_encoder = tf.nn.sigmoid(tf.matmul(H1_encoder,W2_encoder)+B2_encoder)
    H3_encoder = tf.nn.sigmoid(tf.matmul(H2_encoder,W3_encoder)+B3_encoder)
    return H3_encoder 

W1_decoder = tf.Variable(tf.truncated_normal([M,L],stddev=0.1))
B1_decoder = tf.Variable(tf.zeros([L]))

W2_decoder = tf.Variable(tf.truncated_normal([L,K],stddev=0.1))
B2_decoder = tf.Variable(tf.zeros([K]))

W3_decoder = tf.Variable(tf.truncated_normal([K,784],stddev=0.1))
B3_decoder = tf.Variable(tf.zeros([784]))

def decoder( x ):
    H1_decoder = tf.nn.sigmoid(tf.matmul(x,W1_decoder)+B1_decoder)
    H2_decoder = tf.nn.sigmoid(tf.matmul(H1_decoder,W2_decoder)+B2_decoder)
    H3_decoder = tf.nn.sigmoid(tf.matmul(H2_decoder,W3_decoder)+B3_decoder)
    return H3_decoder

y_ = x
output = decoder(encoder(x))
cost = tf.reduce_mean(tf.pow(y_-output,2))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess.run(tf.global_variables_initializer())

# start training

for iteration in range(train_iteration):

    for i in range(total_batch):
        batch = mnist.train_data.next_batch(batch_size)
        xs = batch[0].reshape(-1,784)

        _, c = sess.run((train_step,cost),feed_dict = { x: xs })

        if i % 1000 == 0:
            print("Epoch",i,"cost",c)


