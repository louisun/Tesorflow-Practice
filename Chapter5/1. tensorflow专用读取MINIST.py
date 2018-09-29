# import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# from tensorflow.models.official.mnist import dataset

# 载入 MNIST 数据集
mnist = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True)

# 打印 Training data size: 55000。
print("Training data size: ", mnist.train.num_examples)

# 打印 Validating data size: 5000。
print("Validating data size: ", mnist.validation.num_examples)

# 打印 Testing training data : 10000。
print("Testing training data: ", mnist.test.num_examples)

# 打印 Example traing data: [0. 0. 0. ...]
print("Example training data: ", mnist.train.images[0])

# 打印 Example training data label。
print("Example training data label: ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从 train 的集合中选取 batch_size 个训练数据。
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)