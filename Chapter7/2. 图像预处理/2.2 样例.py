import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
可以通过一张训练图像衍生出很多训练样本，通过对训练样本进行预处理，
训练得的神经网络可以识别不同大小、方位、色彩等方面的实体
'''

# 随机调整图片色彩，定义两种顺序
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)



# 2. 对图片进行预处理，将图片转化成神经网络的输入层数据
def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框
    if bbox is None:
        # 无标准框，认为整个图像都是需要关注的部分[0,0] - [1,1]
        # 为什么要变成 3 维？
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
    # 随机的截取图片中一个块，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    
    # tf.image.sample_distorted_bounding_box 解释：
    # 生成单个随机变形的边界框，输出是可用于裁剪原始图像的单个边框，返回 3 个张量：begin, size, bboxes
    # 前两个 begin, size 用于 tf.slice 裁剪图像，后者用于 tf.image.draw_bounding_boxes 画出边界框

    # images_size 是包含 [height, width, channels] 三个值的一维数组，数值类型必须是 int
    # bounding_boxes 是一个 shape 为 [batch, N, 4] 的三维数组，数据类型为float32
    # batch 是图片批数，N 是有几个边框，4维 是  [y_min, x_min, y_max, x_max] 
    # min_object_covered，默认0.1，表示裁剪区域必须包含所提供的任意一个边界框的至少 min_object_covered 的内容

    distorted_image = tf.slice(image, bbox_begin, bbox_size)


    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))

    # 随机左右反转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用上面的随机色彩函数
    distorted_image = distort_color(distorted_image, np.random.randint(2))

    return distorted_image




# 3. 读取图片
image_raw_data = tf.gfile.FastGFile("../../datasets/cat.jpg", "rb").read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(9):
        # 将图片尺寸调整为 299*299
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()


