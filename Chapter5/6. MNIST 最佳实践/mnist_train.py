'''
模型训练

1. 创建占位符，正则化器

2. 调用前向传播 inference 模块的函数

3. 对 trainable变量滑动平均操作 variables_averages_op

4. 损失(平均交叉熵+正则化项)计算

5. GradientDescentOptimizer 优化器 minimize 损失  train_op

6. 组合 train_op, variables_averages_op

train_op 和 ema_op都是对参数的更新，
但 ema_op 不会直接更新参数变量，而是通过 ema.average 函数返回一个张量，这个张量就是滑动平均后的参数值。
实际应用的过程中，在前向传播的计算流图中用这个张量替代原来的参数变量即可。

---- 以上为一次迭代 ----

for 循环，一共 TRAINING_STEPS 次迭代

获取新的 x,y batch，sess.run(train_op, loss, gloabl_step)

Saver 持久化模型
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./models"
MODEL_NAME="mnist_model"

# 不再将训练和测试跑在一起，只输出 Loss
# 这样可以通过一个单独的测试程序，方便地在滑动平均模型上做测试

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False) # 用于迭代步数计数的变量

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) # 每一次优化 loss 对 global_step 增 1 
    # 组合权重更新和滑动平均操作
    # 在优化方法使用梯度更新模型参数后执行 MovingAverage：
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 获取 global_step 是为了持久化的命名
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # model.ckpt-1000 表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    # mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    mnist = input_data.read_data_sets("/Users/louisun/Projects/MachineLearning/MLPersonalRoot/tf_notes_官方/datasets/MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
