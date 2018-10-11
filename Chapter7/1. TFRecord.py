# TFRecord 文件中的数据都是通过 tf.train.Example PRotocal Buffer 格式存储的
# TFRecord 格式可以统一不同的原始数据格式，更加有效地管理不同的属性

'''
tf.trainExample 的定义:

message Example{
    Features features = 1;
}

message Features {
    map<string, Feature> feature = 1;
};

message Feature{
    oneof kind {
        BytesList bytes_list = 1;
        FlatList float_list = 1;
        Int64List int64_list = 1;
    }
}

tf.train.Example 的数据结构比较简洁，baohan l 
tf.train.Example 包含了一个字典，字符串名称到属性
属性的值可以取字符串(BytesList)，实数列表（FloatList），或者整数列表（Int64List）
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# ----------------- 将输入转换为 TFRecord 格式并保存 -----------------


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式。


def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 读取mnist训练数据。
mnist = input_data.read_data_sets(
    "/Users/louisun/Projects/MachineLearning/MLPersonalRoot/tf_notes_官方/datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels # 0 1 矩阵
pixels = images.shape[1] # 784
num_examples = mnist.train.num_examples

# 输出包含训练数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output.tfrecords") as writer:
    for index in range(num_examples):
        # 创建一个 tf.train.Example
        example = _make_example(pixels, labels[index], images[index])
        # tf.train.Example 序列化为字符串
        writer.write(example.SerializeToString())
print("TFRecord训练文件已保存。")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")





# ----------------- 读取 TFRecord ----------------- 

# 创建一个 TFRecordReader 来读取 TFRecord 文件的
reader = tf.TFRecordReader(

# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["output.tfrecords"])

_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
# tf.parse_single_example 解析指定属性字典的序列化Example
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

# 字符串 tf.decode_raw
images = tf.decode_raw(features['image_raw'], tf.uint8)
# 整数  tf.cast
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# 每次运行会读取 TFRecord 文件中的一个样例，所有读完后，此样例中程序会重头读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])