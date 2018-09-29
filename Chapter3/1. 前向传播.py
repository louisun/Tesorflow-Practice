import tensorflow as tf

# 权重：random_normal 正态分布，stddev 标准差，seed 设定随机种子
# 此外随机生成函数还有 tf.random_uniform 均匀分布，random_gamma Gamma分布
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name="w2")

# 也可以通过常数来初始化变量，如 tf.zeros, tf.ones, tf.fill, tf.constant
# 也可以使用其他变量的初始值来初始化新变量 w_variable.initialized_value()

x = tf.constant([[0.7, 0.9]]) # 1 x 2 矩阵

a = tf.matmul(x, w1) # 1 x 3 
y = tf.matmul(a, w2) # 1 x 1

sess = tf.Session()

# 初始化 w1, w2
# sess.run(w1.initializer)
# sess.run(w2.initializer)

# 当变量数多时，上述运行每个变量的 initalizer 比较麻烦
# 使用全局变量初始化器更方便
init_op = tf.global_variables_initializer()
sess.run(init_op)

print(sess.run(y)) # [[3.957578]]
sess.close()