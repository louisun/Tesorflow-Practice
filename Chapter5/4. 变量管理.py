import tensorflow as tf

# tensorflow 提供了通过变量名来创建和获取一个变量的机制
# 而不需要以参数的形式到处传递，直接通过变量的名字就可以

# 下面两个定义等价
# v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
# v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
# tf.get_variable 会尝试创建一个名为 v 的变量，如果变量存在，将会报错
# 如果想获取某个变量，需要使用 tf.variable_scope 函数建立上下文管理器。


# 在 foo 的命名空间中创建名为 v 的变量
with tf.variable_scope("foo"):
    # 注意 [1] 是 shape
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为 v 已经在 foo 中存在，下面将报错，加上 reuse=True 不再报错。
# 但当设置 reuse=True，tf.variable_scope只能获取已经创建过的变量，若v没创建过会报错
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])


# scope 函数器嵌套测试
with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)  # False
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)  # True
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)  # True
    print(tf.get_variable_scope().reuse)  # False


v1 = tf.get_variable("v1", [1])
print(v1.name) # v1:0
# v1 为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v2", [1])
    print(v2.name) # foo/v2:0

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v3", [1])
        print(v3.name) # foo/bar/v3:0

    v4 = tf.get_variable("v4", [1])
    print(v4.name) # foo/v4:0

# 创建一个名称为空的命名空间，并设置 reuse=True
with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v3", [1])
    # 通过使用带namespace的变量名，来获取其他命名空间下的变量
    print(v5 == v3) # True
    v6 = tf.get_variable("foo/v4", [1])
    print(v6 == v4) # True


# ------- 对之前的前向传播函数进行修改 -------

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
    
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input1')
y = inference(x)

# 需要使用训练好的网络时进行推导时，可以使用 inference(new_x, True)
new_x = ...
new_y = inference(new_x, reuse=True)

# 这样的变量管理方式，就不需要将所有变量作为参数传到不同的函数中
# 当网络结构更复杂、参数更多时，这种变量管理方式，大大提高程序可读性