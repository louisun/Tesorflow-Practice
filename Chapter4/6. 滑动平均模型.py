'''
tf.train.ExponentialMovingAverage 实现滑动平均模型

用途：用于控制变量的更新幅度，使得模型在训练初期参数更新较快，在接近最优值处参数更新较慢，幅度较小
方式：主要通过不断更新衰减率来控制变量的更新幅度

decay 衰减率控制模型更新速度，实际会设置成接近1的数如 0.99
对每个变量维护一个 shadow_variable，初始值即变量初始值
shadow_variable 也称为滑动平均

每次运行变量更新，影子变量的值更新为:
shadow_variable = decay * shadow_variable + (1-decay)*variable

如果 ExponentialMovingAverage 初始化时提供了 num_updates 参数
动态设置 decay 的大小，每次使用的衰减率为：
min{decay, (1+num_updates)/(10+num_updates)}
'''

import tensorflow as tf

# 定义一个计算滑动平均变量，初始值为0，指定变量dtype=float32 必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)

# step 模拟神经网络中迭代轮数，用于动态控制衰减率
step = tf.Variable(0, trainable=False) # trainable=False，Optimizer不会优化此变量

# 滑动平均类，衰减率0.99，控制率的变量 step
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时这个列表的变量都会被更新。
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
  # 初始化所有变量
  tf.global_variables_initializer().run()

  # 通过 ema.average(v1) 获取滑动平均之后变量的取值，在初始化之后变量 v1 的值和 v1 的滑动平均都是0。
  print(sess.run([v1, ema.average(v1)]))

  # 更新变量 v1 的值为 5
  sess.run(tf.assign(v1, 5))

  # 更新 v1 的滑动平均值。衰减率为 min(0.99, (1+step)/(10+step)=0.1)=0.1
  # 所以 v1 的滑动平均会被更新为 0.1*0+0.9*5=4.5
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))

  # 更新 step 的值为 10000.
  sess.run(tf.assign(step, 10000))
  # 更新 v1 的值为 10.
  sess.run(tf.assign(v1, 10))

  # 更新 v1 的滑动平均值，衰减率为 min(0.99, (1+step)/(10+step) ≈ 0.999)=0.99
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))

  # 再次更新滑动平均值，得到的新滑动值为 0.99*4.555+0.01*10=4.60945
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))