#!/usr/bin/python3

from mnist import MnistData
import tensorflow as tf
import numpy as np

pickle_file = 'notMNIST_sanit.pickle'
mnist = MnistData(pickle_file,one_hot=True)

sess = tf.InteractiveSession()

batch_size = 77
total_batch = int(mnist.train_data.data_length/batch_size)
train_iteration = 5
learning_rate = 1e-4

print("total_batch: %d\n" % total_batch)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d_stride1(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def conv2d_stride2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# variable

x = tf.placeholder(tf.float32,[None,28,28,1])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# first conv layer

W_conv1 = weight_variable([6,6,1,6])
b_conv1 = bias_variable([6])

h_conv1 = tf.nn.relu(conv2d_stride1(x,W_conv1) + b_conv1)

# second conv layer

W_conv2 = weight_variable([5,5,6,12])
b_conv2 = bias_variable([12])

h_conv2 = tf.nn.relu(conv2d_stride2(h_conv1,W_conv2) + b_conv2)

# third conv layer

W_conv3 = weight_variable([4,4,12,24])
b_conv3 = bias_variable([24])

h_conv3 = tf.nn.relu(conv2d_stride2(h_conv2,W_conv3)+b_conv3)

# FC layer

W_fc1 = weight_variable([7*7*24,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_conv3,[-1,7*7*24])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

for iteration in range(train_iteration):

    for i in range(total_batch):
        batch = mnist.train_data.next_batch(batch_size)
        xs = np.reshape(batch[0],(-1,28,28,1))
        ys = batch[1]
        sess.run(train_step,feed_dict = { x: xs, y_: ys, keep_prob: 0.75})

        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict = { x: xs, y_: ys, keep_prob: 1.0})
            print("train iteration: %d, train iteration: %d, train accuracy %g" % (iteration,i,train_accuracy))
            print("Cross Entropy ",sess.run(cross_entropy,feed_dict = { x: xs, y_: ys, keep_prob: 1.0}))
            print("test accuracy %g" % accuracy.eval( feed_dict = { x: np.reshape(mnist.test_data.images,(-1,28,28,1)), y_: mnist.test_data.labels, keep_prob: 1.0 }))

            
print("test accuracy %g" % accuracy.eval( feed_dict = { x: np.reshape(mnist.test_data.images,(-1,28,28,1)), y_: mnist.test_data.labels, keep_prob: 1.0 }))



