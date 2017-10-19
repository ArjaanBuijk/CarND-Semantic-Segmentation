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

# 4x4x3 to 8x8x3
h=4
w=4
d=3
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size=2, stride=2, depth=3)

# 1x1x3 to 224x224x3
h=1
w=1
d=3
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size=224, stride=2, depth=3)

# 56x56x3 to 224x224x3
h=56
w=56
d=3
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size=114, stride=2, depth=3)

# 28x28x3 to 224x224x3
h=28
w=28
d=3
x = tf.constant(np.random.randn(1, h, w, d), dtype=tf.float32)
conv = upsample(x, kernel_size=170, stride=2, depth=3)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(conv)

    print('Input Shape: {}'.format(x.get_shape()))
    print('Output Shape: {}'.format(result.shape))
