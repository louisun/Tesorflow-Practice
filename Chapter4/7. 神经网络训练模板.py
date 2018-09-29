import tensorflow as tf

batch_size = n
# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法。
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')

# 定义损失
loss = ...
# 定义优化算法
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 训练
with tf.Session() as sess:
	# 参数初始化。
	...
	# 迭代的更新参数。
	for i in range(STEPS):
		# 准备batch_size个训练数据，一般将所有训练数据随机打乱之后再选取可以得到更好的优化效果。
		current_X, current_Y = ...
		sess.run(train_step,feed_dict={x:current_X,y_:current_Y})