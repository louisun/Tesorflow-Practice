import tensorflow as tf

INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10 

# 通过 tf.get_variable 函数获取变量，在训练时会创建这些变量，在测试时通过保存的模型加载这些变量
# 更为方便的是，可以在变量加载时将滑动平均变量重命名，可以直接通过同样名字在训练时使用变量自身
# 而在测试时使用变量的滑动平均值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2