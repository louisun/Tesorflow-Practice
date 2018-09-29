# 注：此代码段不可运行

import tensorflow as tf

a = tf.nn.relu(tf.matmul(x,w1) + biases1)
b = tf.nn.relu(tf.matmul(a,w2) + biases2)