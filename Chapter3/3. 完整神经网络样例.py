# 训练神经网络的过程可以分为以下 3 个步骤
# 1. 定义神经网络的结构和前向传播的输出结果
# 2. 定义损失函数以及选择反向传播优化的算法
# 3. 生成会话(tf.Session)并且在训练数据上反复运行反向传播优化算法

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 在 shape 的一个维度上使用 None 可以自定义用多少组数据
# 在训练时用较小的 batch, 在测试时用所有数据。
# 当数据集较小时这样比较方便测试，但数据集较大时，可能会导致内存溢出。
x = tf.placeholder(tf.float32,shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1), name='y-input') # label

# 定义神经网络前向传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2) # predict

# 定义损失函数和反向传播算法
y = tf.sigmoid(y)

# 定义损失函数来刻画预测值与真实值之间的差距
# clip_by_value(tensor, min, max) 限制张量范围
# 交叉熵公式，书 p75
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

# 反向传播，优化权重降低损失
# 0.001 是学习率
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2) # 128 x 2 的训练集

# 定义规则来给出样本的标签
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行程序
with tf.Session() as sess:
  # 初始化所有变量，但没有运行
  init_op = tf.global_variables_initializer()

  # 现在运行初始化的变量
  sess.run(init_op)
  print("w1:", sess.run(w1))
  print("w2:", sess.run(w2))

  # 设定训练的轮数
  STEPS = 5000
  for i in range(STEPS):
    # 每次选取 batch_size 个样本进行训练
    start = (i * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)

    # 通过选取的样本训练神经网络并更新参数
    sess.run(train_step,feed_dict={x:X[start:end], y_:Y[start:end]})

    if i % 1000 == 0:
      # 每隔一段时间计算所有数据的交叉熵
      total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
      print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

  print("w1:", sess.run(w1))
  print("w2:", sess.run(w2))