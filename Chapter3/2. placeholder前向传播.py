import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name="w2")

# ---------------------------------------------------------
# 为什么要引入占位符？
# 防止图节点过多，利用率低，占位符的数据在程序运行时指定

# placeholder的shape不一定要定义，但固定维度能降低出错概率
x = tf.placeholder(tf.float32, shape=(1,2), name="input")
# ---------------------------------------------------------

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)

# ---------------------------------------------------------
# 占位符需要 feed_dict 字典
print(sess.run(y, feed_dict={x: [[0.7, 0.9]]})) # [[3.957578]]
# ---------------------------------------------------------
sess.close()