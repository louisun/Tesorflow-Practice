import tensorflow as tf


slim = tf.contrib.slim

# slim.arg_scope 函数用于设置默认的参数取值, 第一个参数是一个函数列表，
# 这个函数列表中的函数将使用默认的参数取值，比如通过下面的定义，调用 slim.conv2d(net, 320, [1, 1])
# 函数会自动加上 stride = 1 和 padding = 'SAME' 的参数
# 如果在函数调用时指定了 stride ，那么这里设置的默认值就不会再使用
# 通过这种方式可以进一步减少冗余代码。

with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    # 此处省略了 Inception-v3 模型中其他的网络结构而直接实现最后面框中的 Inception 结构。
    # 假设输入图片经过的神经网络前向传播的结果而保存在变量 net 中。
    net = "上一次的输入节点矩阵"
    # 为一个 Inception 模块声明一个统一的变量命名空间
    with tf.variable_scope('Mixed_7c'):

        # 为 Inception 模块每条路径声明一个命名空间
        # 第 1 条路径
        with tf.variable_scope('Branch_0'):
            # 实现一个过滤器边长为 1，深度为 320 的卷积层。
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

        # 第 2 条路径, 这条计算路径上的结构本身也是一个 Inception 结构。
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            # tf.concat 函数可以将多个矩阵拼接在一起。tf.concat 函数的第一个参数指定了拼接的维度，
            # 这里给出的"3"代表了矩阵在深度这个维度上进行拼接
            branch_1 = tf.concat(3, [
                slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0a_1x3'),
                slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0a_3x1')
            ])

        # 第 3 条路径
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d( net, 448, [1, 1], scope='Conv2d_0a_1x1')

            branch_2 = slim.conv2d( branch_2, 384, [1, 3], scope='Conv2d_0a_1x3')

            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0a_1x3'),
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0a_3x1')
            ])

        # 第 4 条路径。
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(
                branch_3, 192, [1, 1], scope='Conv2d_0a_1x1')
        # 当前 Inception 模块的最后输出是由上面四个计算结果拼接得到的。
        # 这里的 3 表示在第三维度上进行连接。
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
