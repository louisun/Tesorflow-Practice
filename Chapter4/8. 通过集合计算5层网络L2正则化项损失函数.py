import tensorflow as tf

# 获取一层神经网络边上的权重，并将这个权重的 L2 正则化损失加入名称为 'loss' 的集合中。
def get_weight(shape, var_lambda):
  var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
  # add_to_collection 函数将这个新生成变量的 L2 正则化损失项加入集合。
  # 第一个参数 ‘loses’ 是集合的名字，第二个参数是要加入这个集合的内容。
  tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(var_lambda)(var))
  return var

x = tf.placeholder(tf.float32,shape=(None, 2))
y_ = tf.placeholder(tf.float32,shape=(None, 1))
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 5, 3, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
cur_layer = x
in_dimension = layer_dimension[0]
# 通过一个循环来生成5层全连接的神经网络结构。
for i in range(1, n_layers):
  out_dimension = layer_dimension[i]
  weight = get_weight([in_dimension, out_dimension], 0.003)
  bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
  # 使用 ReLU 激活函数
  # 视同 relu 将是折线， elu 将是平滑的线
  cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+bias)
  in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的 L2 正则化损失加入到图上的集合
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# tf.get_collection 返回集合元素列表
# tf.add_n 将列表中的元素加起来就可以得到最终想要的损失函数
loss = tf.add_n(tf.get_collection('losses'))