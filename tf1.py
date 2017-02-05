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
batch_size = 100
training_iteration = 30
display_step = 2

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

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

Y1 = tf.nn.relu(tf.matmul(x,W1)+B1)
Y2 = tf.nn.relu(tf.matmul(Y1,W2)+B2)
Y3 = tf.nn.relu(tf.matmul(Y2,W3)+B3)
Y4 = tf.nn.relu(tf.matmul(Y3,W4)+B4)
Y4D = tf.nn.dropout(Y4,keep_prob)
logits = tf.matmul(Y4D,W5)+B5

with tf.name_scope("cost_function") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
    tf.scalar_summary("cost_function", cross_entropy)

with tf.name_scope("training") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

merged_summary_op = tf.merge_all_summaries()

sess.run(tf.global_variables_initializer())

summary_writer = tf.train.SummaryWriter('./tf1_summary',graph_def=sess.graph_def)

for iteration in range(training_iteration):
    avg_cost = 0.
    total_batch = int(mnist.train_data.data_length / batch_size)

    for i in range(total_batch):
        batch = mnist.train_data.next_batch(batch_size)
        batch_xs = np.reshape(batch[0],(-1,784))
        batch_ys = batch[1]
        sess.run(train_step, feed_dict = { x: batch_xs, y_: batch_ys, keep_prob: 0.75 } )
        
#       avg_cost += sess.run(cross_entropy, feed_dict = { x: batch_xs, y_: batch_ys })/total_batch
#        summary_str = sess.run(merged_summary_op, feed_dict = { x: batch_xs , y_:batch_ys})
#       summary_writer.add_summary(summary_str, iteration * total_batch + i )

    if iteration % display_step == 0:
        print("Iteration:",'%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

print("training complete")

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy, feed_dict = { x: np.reshape(mnist.test_data.images,(-1,784)), y_: mnist.test_data.labels, keep_prob: 1}))

sess.close()
