import tensorflow as tf
filter_weight = tf.get_variable(
    'weights', [5,5,3,16], # 5*5 过滤器， 3 层(rgb)，16 层深度（过滤器个数）=下一层矩阵深度
    initializer=tf.truncated_normal_initializer(stddev=1)
)

# 由于 filter 的深度是 16，bias 的深度也是16
biases = tf.get_variable(
    'biases', [16],
    initializer=tf.constant_initializer(0.1)
)


# input[1,:,:,:] 表示第一张图片
# filter_weight 是 4 维的权重：[h, w, c, n]
# strides 是各维度上的步长，四维数组 [1,h,w,1]
# 但第一个数和最后一个数要求一定是 1，因为卷积层步长只对矩阵的高h和宽w有效

# padding 是填充方法：'SAME' 表示全 0 填充，'VALID' 表示不添加
conv = tf.nn.conv2d(
    input, filter_weight, strides=[1,1,1,1], padding='SAME'
) 


# bias_add 方便地添加偏置项
bias = tf.nn.bias_add(conv, biases)

# 通过 Relu 函数去线性化
actived_conv = tf.nn.relu(bias)


# 池化层
# 池化层也是通过一个类似过滤器的结构，只是它不求加权和，而是采用最大值或平均值等运算
# 而且卷积层的过滤器是横跨整个深度的（比如rgb 3 层一起求加权），而池化层只针对一个深度
# 因此池化层的过滤器，除了在长和宽维度上移动，还需要在深度维度上移动

# 最大池化层
# ksize 提供过滤器尺寸 [1,h,w,1]），只针对单输入样例、单深度
# 而 conv2d 的 weight 维度是 [h, w, c, n]，注意和 ksize 维度的区别
# 实际 ksize 用的最多的是 [1,2,2,1] 和 [1,3,3,1]

# strides 提供步长信息, padding 指示是否全0填充
pool_max = tf.nn.max_pool(actived_conv, ksize=[1,3,3,1],  
    strides=[1,2,2,1], padding='SAME'
)

# 平均池化层
pool_avg = tf.nn.max_pool(actived_conv, ksize=[1,3,3,1], 
    strides=[1,2,2,1], padding='SAME'
)