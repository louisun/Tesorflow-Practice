# 注：此代码段不可运行

# 交叉熵公式：H(p, q) = -SUM(p(x) * log(q(x)))
# 交叉熵刻画两个概率分布之间的距离，表示：通过概率分布 q 来表达概率分布 p 的困难程度
# 交叉熵越小，两个概率分布越接近
# softmax 能够把输出变成一个概率分布

import tensorflow as tf

# y_: 真实值(0,1)
# y:预测值

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# cross_entropy 和 softmax 的封装
# logits:  无尺度化的 log 概率
# labels:  每个条目 labels[i] 必须是[0, num_classes) 的索引或者-1. 如果是-1，则相应的损失为0，不用考虑 logits[i] 的值。
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

# 对于回归问题，最常用的损失函数是 均方误差 MSE (mean squared error)
mse = tf.reduce_mean(tf.square(y_ - y))

