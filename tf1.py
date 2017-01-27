#!/usr/bin/python3

from mnist import MnistData
import tensorflow as tf
import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

K = 200
L = 100
M = 60
N = 30
learning_rate = 0.003

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

W1 = tf.Variable(tf.truncated_normal([784,K],stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K,L],stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L,M],stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M,N],stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N,10],stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

Y1 = tf.nn.relu(tf.matmul(x,W1)+B1)
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+B2)
Y3 = tf.nn.relu(tf.matmul(Y2,W3)+B3)
Y4 = tf.nn.relu(tf.matmul(Y3,W4)+B4)
keep_prob = tf.placeholder(tf.float32)
Y4D = tf.nn.dropout(Y4,keep_prob)
logits = tf.matmul(Y4D,W5)+B5

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

for i in range(10000):
    batch = mnist.train_data.next_batch(100)

    a = np.reshape(batch[0],(-1,784))
    print("shape:",a.shape,batch[1].shape)
    
    #sess.run(train_step,feed_dict= { x: np.reshape(batch[0],(-1,784)), y_: batch[1] })
    





