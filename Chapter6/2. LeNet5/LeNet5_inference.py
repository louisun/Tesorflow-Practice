import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

# train 参数 True/False 用于区分训练和测试阶段
def inference(input_tensor, train, regularizer):
    # 通过变量命名空间区分不同层
    with tf.variable_scope('layer1-conv1'):
        # 权重, bias`
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # conv
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # relu
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases)) # tf.nn.bias_add

    with tf.name_scope("layer2-pool1"):
        # pool filter 是 2*2，长、宽步长都是2
        # ksize 提供过滤器尺寸 [1,h,w,1]），只针对单输入样例、单深度
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        # 注意 conv2 权重第3个位置(原先chanel位置)为 conv1 权重的深度
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层的输出要转为全连接层的输入
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.shape(pool2) 返回的是Tensorflow里面的张量，而 pool.get_shape() 返回的是元祖 TensorShape([Dimension(?), Dimension(7), Dimension(7), Dimension(64)])
        # 注意每层神经网络的输入输出都是一个 batch 的矩阵，所以第一个维度（下标为0）包含的是每批 batch 数据的个数(多少个样本) 
        pool_shape = pool2.get_shape().as_list() # 将上面的元祖转为真正的数值 shape  [batch_num, 7,7,64]
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) # 每个样本的全连接向量

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加正则化
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 只在全连接 训练时，使用 dropout，以减轻过拟合
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        # logit 通过 softmax 知乎就得到了分类结果

    return logit