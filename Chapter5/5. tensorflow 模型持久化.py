# tf.train.Saver 保存计算图

import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1+v2

saver = tf.train.Saver()

# ----------- 持久化：保存图结构和参数 -----------
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 将模型保存到 ./model/model.ckpt
    saver.save(sess, "./models/model.ckpt")
    # 除了.ckpt文件外，还有 3 个文件（图的结构和图上参数取值分开保存）
    # model.ckpt.meta 保存计算图结构
    # model.ckpt 保存每一个变量的取值
    # checkpoint 文件保存一个目录下所有模型文件列表

# ----------- 加载恢复持久化的图和数据 -----------
# 加载、恢复图
saver = tf.train.import_meta_graph(
    "./models/model.ckpt.meta"  # 注意是 meta 即图结构文件
)
# 也可以再定义一遍

with tf.Session() as sess:
    # 恢复数据
    saver.restore(sess, "./models/model.ckpt")  # .ckpt 恢复数据文件
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    # [3.]

# ------------ 保存、加载部分变量 -----------
# 比如有个之前训练好的五层神经网络模型，想尝试一个六层网络
# 可以将前5层的参数直接加载到新模型，而仅仅将最后一层网络重新训练
saver.train.Saver([v1])  # 只有变量 V1 被加载尽量
# 直接运行会报错，因为v2没被加载，运行初始化之前没有值


# ----------- 变量重命名 -----------

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v2")

# 原来名称为"v1"的变量，可以加载到现在名称为“other-v1”的v1变量中
saver = tf.train.Saver({"v1": v1, "v2"})


# ----------- 保存滑动平均模型 -----------

import tensorflow as tf
v = tf.Variable(0, dtype=tf.float32, name="v")

for variables in tf.global_variables():
    print(variables.name)  # v:0

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())

# 声明滑动平均模型之后，Tensorflow会自动生成一个影子变量 v/ExponentialMovingAverage
for variables in tf.global_variables():
    # v:0 和 v/ExponentialMovingAverage
    print(variables)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    saver.save(sess, "./models/model.ckpt")
    print(sess.run([v, ema.average(v)]))  # [10.0, 0.099999905]

# ----------- 恢复滑动平均值 -----------

v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量 v 的滑动平均值直接赋值给 v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})

with tf.Session() as sess:
    # 将滑动平均值(shadow)取值恢复到变量 v
    saver.restore(sess, "./models/emaModel.ckpt")
    print(sess.run(v))


# 为了方便加载时重命名滑动平均变量
# tf.train.ExponentialMovingAverage 类提供了 variables_to_restore 函数
# 来生成 tf.train.Saver 类所需要的变量重命名字典
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

# 通过使用 variables_to_restore 函数可以直接生成上面代码提供的字典
print(ema.variables_to_restore())
# 输出{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "./models/emaModel.ckpt")
    print(sess.run(v))
    # 输出 v 的滑动平均值


# ------------ 只保存常量 ------------
# 是时候只需要保存变量及其取值，而不需要模型结构、变量初始化、辅助节点等其他信息
# 保存到 pb 文件

import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1+v2

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 导出当前计算图的 GraphDef 部分，
    # 只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将图中的变量及取值化为常数，同时将图中不需要的节点去掉。
    # 如果只关心程序中定义的某些计算，和这些计算无关的节点就没有必要导出并保存了。
    # 在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。add节点是上面定义的两个变量相加的
    # 操作。注意这里给出的是计算节点的名称，所以没有后面的:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, [
                                                                 'add'])
    # 将导出的模型存入文件。
    with tf.gfile.GFile("./models/combined_model.pb", "wb") as f:
    f.write(output_graph_def.SerializeToString())

# 恢复

with tf.Session as tf:
    model_filename = "./models/combined_model.pb"
    # 读取保存的模型文件，将文件解析成对应的 GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # 将 graph_def 中保存的图加载到当前图中
        # return_elements=["add:0"] 给出返回的张量名称，而在保存时候给出的是计算节点的名称，所以为“add”
        # 在加载的时候时候给出的是张量名称，所以是 add:0
        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print(sess.run(result)) # [3.0]

