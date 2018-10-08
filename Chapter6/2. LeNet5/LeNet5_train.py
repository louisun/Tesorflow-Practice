import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_inference
import os
import numpy as np

# 网络相关参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99


MODEL_SAVE_PATH="./models"
MODEL_NAME="lenet5_model"

def train(mnist):
    # 输入为 4 维矩阵的 placeholder：x (n, h, w, c)
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        LeNet5_inference.IMAGE_SIZE,
        LeNet5_inference.IMAGE_SIZE,
        LeNet5_inference.NUM_CHANNELS],
        name='x-input')

    # label
    y_ = tf.placeholder(
        tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播获取 预测值 logit
    y = LeNet5_inference.inference(x, True, regularizer)  # 第二个参数表示是否是训练阶段

    global_step = tf.Variable(0, trainable=False) # 用于迭代步数计数的变量

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # tf.argmax(y_,1) 才是预测的标签
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.NUM_CHANNELS))

            # feed 的时候要 reshape 为 4维
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={
                                           x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (
                    step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    # mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    mnist = input_data.read_data_sets("/Users/louisun/Projects/MachineLearning/MLPersonalRoot/tf_notes_官方/datasets/MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
