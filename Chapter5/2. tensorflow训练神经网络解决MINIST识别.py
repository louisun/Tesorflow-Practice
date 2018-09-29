import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入层节点个数，相当于图片像素数 28*28
OUTPUT_NODE = 10  # 10 类数字

LAYER1_NODE = 500 # 隐藏层节点数（设该网络只有一个隐藏层）
BATCH_SIZE = 100  # 每次batch打包的样本个数      

# 弄清楚学习率衰减 & 滑动平均的区别
LEARNING_RATE_BASE = 0.8 # 基础学习率
LEARNING_RATE_DECAY = 0.99 # 学习衰减率

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000 # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率

# 辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
# 这是一个使用 Relu 激活层的三层全连接神经网络
# 支持传入用于计算参数平均值的类，方便在测试时使用滑动平均模型
# (input) --w1-- (hidden1) --relu--w2--(output)
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 是否提供滑动平均类，若无则直接使用参数当前取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用 avg_class.average 函数计算出变量的滑动平均值
        # 其实就是对 w1, w2 使用其滑动平均值

        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + 
            avg_class.average(biases1)
        )
        

        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)



def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    # 截断的正态分布: 产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    # 和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值
    # 所以指定 trainable=False，一般都会将代表训练轮数的变量指定为不可训练的参数
    # 这样就不会加到 trainable_variables() 集合中
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )

    # 在所有 “代表神经网络参数” 的变量上 "使用滑动平均"，其他辅助变量如 global_step 就不需要
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables() # 返回 GraphKeys.TRAINABLE_VARIABLES 中的元素
    ) 
    # 到底什么是滑动平均？让这些参数在训练初期更新快，在接近最优值处更新慢，动态控制参数的更新幅度
    
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2
    )

    # 交叉熵作为预测值和真实值差距的损失函数，当正确答案只有一个时，可以使用此函数快速计算
    # logits 是预测值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, axis=1) # 第二个参数 1 是 axis, 注意 y_ 的维数目
    )

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 正则化器，传入系数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 正则化项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失
    loss = cross_entropy_mean + regularization # 注意 loss 是 cross_entropy 的均值 + 正则化项

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                    # 基础学习率
        global_step,                           # 当前迭代轮数
        mnist.train.num_examples / BATCH_SIZE, # 过完所有训练数据需要的迭代次数
        LEARNING_RATE_DECAY                    # 学习衰减率
    )



    # -------------- 优化算法 --------------
    # 每次迭代，会对global_step +1
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )

    # 训练时，每过一遍数据既要通过反向传播来更新神经网络的参数，又要更新每一个参数的滑动平均值
    # 为了一次完成多个操作，Tensorflow 提供了 tf.control_dependencies 和 tf.group 两种操作
    # 下面程序和 
    # train_op =  tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train') # no_op；什么也不做，只是返回这个控制依赖的 op 
    
    # 正确预测的布尔数组
    correct_prediction = tf.equal(tf.argmax(average_y, axis=1), tf.argmax(y_, axis=1)) # True or False
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run() # sess.run(this_initializer) 也是一样的

        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 在真实的应用中，这部分数据在训练时是不可见的，只作为模型优劣的最后评价标准
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            # 每 100 轮输出一次在验证集上的测试结果
            if i % 1000 == 0:
                # 因为 MNIST是数据集较小，一次可以处理所有验证数据，为了计算方便，并没有将验证集氛围更小的batch
                # 当网络更复杂或验证集太大时，太大的 batch 会导致计算时间过长或内存溢出
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE) # 从 mnist 训练集中获取下一 batch 的数据用于训练
            sess.run(train_op, feed_dict={x:xs, y_:ys}) # 注意：所有训练过程都是在 train_op 中（包括权值优化、滑动平均）

        # 所有训练轮数结束后，在测试集上验证最终的正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g " % (TRAINING_STEPS, test_acc))

def main(argv=None):
    # mnist = input_data.read_data_sets("/tmp/data", one_hot=True) # 第一个参数是存放数据的位置，采用 one_hot 编码
    mnist = input_data.read_data_sets("/Users/louisun/Projects/MachineLearning/MLPersonalRoot/tensorflow_notes/datasets/MNIST_data/", one_hot=True)
    
    train(mnist)

# Tensorflow 提供主程序入口：tf.app.run() 会自动调用 main 函数
if __name__ == '__main__':
    tf.app.run()