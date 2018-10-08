'''
验证训练得到的模型的准确率

定义默认图，创建 x, y 占位符，获取验证集
利用模型前向传播

从每个持久化的模型中恢复参数的滑动平均值（而不是原先值）
预测结果，获得准确率
'''
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_inference
import LeNet5_train


# 每 10 秒加载一次最新的模型，并在验证集上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            None,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.IMAGE_SIZE,
            LeNet5_inference.NUM_CHANNELS],
            name='x-input')

        y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE], name='y-input')
        # 在整个验证集上验证
        validate_feed = {
            x: mnist.validation.images.reshape(
                None,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.IMAGE_SIZE,
                LeNet5_inference.NUM_CHANNELS), 
            y_: mnist.validation.labels
        }
        
        # 预测 y，计算准确率
        y = LeNet5_inference.inference(x, False, None) # 测试时，train=False
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名方式来加载模型，在前向传播的过程中不需要调用求滑动平均的函数(apply)来获取平均值
        # 所以这里用于计算正则化损失的函数被设置为 None 
        variable_averages = tf.train.ExponentialMovingAverage(LeNet5_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore() # 把我们的滑动平均值恢复到当前变量
        # {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
        saver = tf.train.Saver(variables_to_restore) # 指定要从持久化模型中恢复的变量

        # 每隔 EVAL_INTERVAL_SECS 秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # get_checkpoint_state 会自动从 checkpoint 文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(LeNet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path) # 从该ckpt加载模型，然后计算该 global_step 的准确度
                    # for v in tf.global_variables():
                    #     print(v.name, ":", v.eval())
                    # print("#####################")    
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # 得到模型保存时迭代的轮数
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed) # 得到该轮模型在验证集上的准确率
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    # mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    mnist = input_data.read_data_sets("/Users/louisun/Projects/MachineLearning/MLPersonalRoot/tf_notes_官方/datasets/MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()


# ------------- 查看已经保存到第几轮了 ---------------
# import tensorflow as tf
# ckpt = tf.train.get_checkpoint_state("MNIST_model/")
# ckpt.model_checkpoint_path
# 'MNIST_model/mnist_model-9001'