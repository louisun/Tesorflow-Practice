# 流程图：
# https://ws3.sinaimg.cn/large/006tNbRwly1fw4cyop5s2j30uo0fm1kc.jpg


import tensorflow as tf

files = tf.train.match_filenames_once("/path/to/file_pattern-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string), # 存储图像原始数据
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }
)

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# 从原始图像解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

# 还是用这本章 2.2 样例.py 中的图像预处理函数 preprocess_for_train

image_size = 299 # 神经网络输入层图片大小
distorted_image = preprocess_for_train(
    decoded_image, image_size, image_size, None
)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=min_after_dequeue
)


# 定义神经网络的结构和优化过程， image_batch 可以作为输入提供给神经网络的输入层
# 这里引用之前的函数
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # 神经网络训练准备工作，包括变量初始化，线程启动
    sess.run((tf.global_variables_initializer, tf.local))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # 神经网络训练过程
    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)
    
    coord.request_stop()
    coord.join()



