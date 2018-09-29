# 注：此代码段不可运行
import tensorflow as tf

w = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w)

# L2 正则化的 loss
# y_ 为真值
# tf.contrib.layers.l2_regularizer() 传入 lambda 系数，返回一个函数

loss = tf.reduce_mean(tf.square(y_ - y) + 
       tf.contrib.layers.l2_regularizer(lambda_)(w))

