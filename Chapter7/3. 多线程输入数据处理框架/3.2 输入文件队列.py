# ---------------------- 1. 生成文件存储的样例数据 ----------------------

import tensorflow as tf

# 创建 TFRecord 文件的帮助函数


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况，将数据写入不同的文件
num_shards = 2  # 共跌入多少文件
instances_per_shard = 2  # 定义每个五年级有多少数据

for i in range(num_shards):
    # 将数据分为多个文件时，可以将不同文件以类似 0000n-of-0000m 的后缀区分。
    filename = ('data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)

    # 将数据封装成 Example 结构并写入 TFRecord 文件。
    for j in range(instances_per_shard):
        # Example 结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)
        }))

        writer.write(example.SerializeToString())
    writer.close()


# ---------------------- 2. 读取文件 ----------------------

files = tf.train.match_filenames_once("data.tfrecords-*")

# 将 shuffle 设置为 false ，避免随机打乱读文件的顺序，在真实问题中将设置成 True
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 从输入文件队列里读取文件

features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }
)

with tf.Session() as sess:
    # tf.tarin.match_filenames_once 需要初始化一些变量 tf.local.variables_initializer().run()
    tf.local_variables_initializer().run()
    print(sess.run(files))

    # 声明 Coordinator 类来协同不同的线程。
    coord = tf.train.Coordinator()
    # 关于 start_queue_runners
    # 只是一个组合操作 to add_queue_runner()，启动线程 for 所有的图中收集的 queue runners 
    # 返回线程列表
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 多次执行获取数据的操作。
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)

print("\n\n")

# ---------------------- 3. 组合训练数据(batching) ----------------------
# 即将多个输入样例组织为一个 batch

# i 代表一个样例的特征向量， j 表示样例对应的标签
example, label = features['i'], features['j']

# 一个 batch 的样例数
batch_size = 2

# 组合样例队列最多可以存储的样例个数，太大占内存，太小易阻塞降低效率
capacity = 1000 + 3 * batch_size

# 使用 tf.train.batch 组合样例， [example, label] 参数给出了需要组合的元素
# capacity 表示队列最大容量

example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity
) # batch 也是用了 QueueRunner


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取并打印组合后的样例，在真实问题中，这个输出一般会被作为神经网络的输入
    print("example_batch   label_batch")

    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch]
        )
            
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)

# 如果不想按顺序组成 batch，可以使用 tf.train.shuffle_batch 
# min_after_deque 限制了出队时，队列中最小的元素数
# 因为元素太少，shuffle 就没有用了
# tf.train.shuffle_batch([example,label], batch_size=batch_size, capacity=capacity, min_after_deque=30)