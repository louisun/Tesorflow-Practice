'''
新框架中，每一个数据来源被抽象为一个“数据集”
可以以数据为基本对象，方便地进行 batching, shuffle 等操作
数据集框架：tf.data 
'''


# ------------------ 1. from_tensor_slices 从数组创建数据集 ------------------ 
import tensorflow as tf

# 从数组创建数据集
input_data = [1, 2, 3, 5, 8]
# 从一个 Tensor 返回一个 dataset
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# 定义迭代器
iterator = dataset.make_one_shot_iterator()

# get_next() 返回代表一个输入数据的张量。
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
# 1
# 4
# 9
# 25
# 64


# 利用数据集有 3 个方法
# 1. 定义数据集的构造方法
# 2. 定义遍历器
# 3. 使用 get_next 从遍历器中读取张量

# 在真实项目中，训练数据通常放在磁盘文件上
# 比如在自然语言处理的任务中，训练数据通常是每行一条
# 这时可用 TextLineDataset来方便地读取数据


# ------------------ 2. TextLineDataset 行文件数据集------------------ 
print('\n\n')
# 创建文本文件作为本例的输入。
with open("./test1.txt", "w") as file:
    file.write("File1, line1.\n") 
    file.write("File1, line2.\n")

with open("./test2.txt", "w") as file:
    file.write("File2, line1.\n") 
    file.write("File2, line2.\n")

# 从文本文件创建数据集, 这里可以提供多个文件。
input_files = ["./test1.txt", "./test2.txt"]
dataset = tf.data.TextLineDataset(input_files)

# 定义迭代器。
iterator = dataset.make_one_shot_iterator()

# 这里get_next()返回一个字符串类型的张量，代表文件中的一行。
x = iterator.get_next()  
with tf.Session() as sess:
    for i in range(4):
        print(sess.run(x))

# b'File1, line1.'
# b'File1, line2.'
# b'File2, line1.'
# b'File2, line2.'


# ------------------ 3. TFRecordDataset 数据集------------------ 
print('\n\n')

# 解析一个TFRecord的方法。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8) # 从string解码
    retyped_images = tf.cast(decoded_images, tf.float32) # 整数类型再转换为实数
    images = tf.reshape(retyped_images, [784]) # 调整大小
    labels = tf.cast(features['label'],tf.int32)
    #pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels

# 从TFRecord文件创建数据集。这里可以提供多个文件。
input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)

# dataset.map 传入 parser 函数
# map()函数表示对数据集中的每一条数据进行调用解析方法
dataset = dataset.map(parser)

# 定义遍历数据集的迭代器。
iterator = dataset.make_one_shot_iterator()

# 读取数据，可用于进一步计算
image, label = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label]) 
        print(y)