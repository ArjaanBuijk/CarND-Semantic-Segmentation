#import oldtensorflow as tf
import tensorflow as tf
import numpy as np


def upsample(x, kernel_size=2, stride=2, depth=3):
    """
    Apply a two times upsample on x and return the result.
    :x: 4-Rank Tensor
    :return: TF Operation
    """
    return tf.layers.conv2d_transpose(x, d, kernel_size, stride)


# 1x1x3 to 4x4x3
h=1
w=1
d=3
kernel_size=4
stride=2
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size, stride, depth=d)

# 1x1x3 to 10x10x3
h=4
w=4
d=3
kernel_size=4
stride=2
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size, stride, depth=d)

# 10x10x3 to 88x88x3
h=10
w=10
d=3
kernel_size=16
stride=8
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size, stride, depth=d)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(conv)

    print('Input Shape: {}'.format(x.get_shape()))
    print('Output Shape: {}'.format(result.shape))
