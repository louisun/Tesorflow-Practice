# 注: 该代码段无法运行
import tensorflow as tf

STEPS = 100
global_step = tf.Variable(0)

# tf.train.exponential_decay 生成指数级减小的学习率，此函数实现了：
# decayed_learning_rate = learning_rate * 
#                         decay_rate ^ (global_step / decay_step) 

# decay_rate 为衰减系数，decay_steps 为衰减速度
# staircase 默认为 False 代表曲线， Ture 表示阶梯状
# staircase 为 True 时，是对 global_step/decay_step取整
# 这种情况下，decay_step 往往是遍历一遍数据需要的迭代轮数
# 这个迭代轮数 = 总样本 / batch_size
# 常见的场景就是，每完整过完一遍训练数据，学习率就减小一次

# decay every 100000 steps with a base of 0.96
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

# 在Optimizer中传入指数衰减的 learning_rate
# 在 minimize 中传入 global_step 将自动增加 global_step，学习率也得到相应更新
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)