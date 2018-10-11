
'''
tensorflow 中队列不仅是一种数据结构，还是异步计算张量取值的一个重要机制
'''
import tensorflow as tf


#  --------------- 1. 创建队列，操作里面的元素 -------------------


q = tf.FIFOQueue(2, "int32")
# 除了 FIFOQueue，还有一种 RandomShuffleQueue，会将队列元素打乱，每次出队都随机

init = q.enqueue_many(([0, 10],))

x = q.dequeue()
y = x+1
# 入队操作
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)

# 0
# 10
# 1
# 11
# 2


#  --------------- 2. tf.Coordinator 协同线程一起停止-------------------

# tf.train.Coordinator 和 tf.train.QueueRunner 两个类来完成多线程协同功能
# tf.train.Coordinator 主要用于协同多个线程一起停止
# 提供了 should_stop, request_stop, join 三个函数
# should_stop 返回 True 表示当前线程一进退出
# 每个线程通过 request_stop 来通知其他线程退出
# 当一个线程 request_stop 后，其 should_stop 返回值为 True
# 这样其他线程就可以同时终止了
import numpy as np
import threading
import time


# 程序每嗝 1s 判断是否需要停止并打印自己的 ID
def MyLoop(coord, worker_id):
    # Coordinator 类提供的 should_stop() 来判断当前线程是否需要停止
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: %d" % worker_id)
            # 通知其他线程停止
            coord.request_stop()
        else:
            print("Working on id: %d" % worker_id)
        time.sleep(1)


coord = tf.train.Coordinator()

# 创建 5 个线程(使用同一个 Coordinator)
threads = [
    threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)
]

# 启动所有线程
for t in threads:
    t.start()

# 等待所有线程退出
coord.join(threads)


#  --------------- 3. tf.QueueRunner 启动多个线程操作同一队列 -------------------

import tensorflow as tf


queue = tf.FIFOQueue(100, "float")

enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用 tf.train.QueueRunner 来创建多个线程运行队列的入队操作
# QueueRunner 的第一个参数给出了被操作的队列， [enqueue_op]*5
# 表示需要启动 5 个线程，每个线程中运行的是 enqueue_op 操作。
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)

# 将定义过 QueueRunner 加入 TensorFlow 计算图上指定的集合。
# add_queue_runner 函数没有指定集合，则加入默认集合 QUEUE_RUNNERS 。
# 下面的函数就是将刚刚定义的 qr 加入默认的 QUEUE_RUNNERS 集合。
tf.train.add_queue_runner(qr)
# 定义出队操作。
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用 tf.train.Coordinator 来协同启动的线程。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    # 使用 tf.train.Coordinator 来停止所有的线程
    coord.request_stop()
    # 等待所有线程退出。
    coord.join(threads)
