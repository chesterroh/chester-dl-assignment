#!/usr/bin/python3

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

K = 200
L = 100
M = 60
N = 30

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

W1 = tf.Variable(tf.truncated_normal([28*28,K],stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.zeros([N,10]))
B5 = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

Y1 = tf.nn.relu(tf.matmul(x,W1)+B1)
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+B2)
Y3 = tf.nn.relu(tf.matmul(Y2,W3)+B3)
Y4 = tf.nn.relu(tf.matmul(Y3,W4)+B4)
Y4D = tf.nn.dropout(Y4,0.75)
y = tf.matmul(Y4D,W5)+B5

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))

optimizer = tf.train.GradientDescentOptimizer(0.002)
train_step = optimizer.minimize(cross_entropy)

for i in range(200000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict = { x: batch[0], y_: batch[1] })

    if i % 1000 == 0:
        print("Epoch :",i)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print(accuracy.eval(feed_dict= { x: mnist.test.images, y_: mnist.test.labels}))


