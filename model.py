from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from keras.layers.advanced_activations import LeakyReLU, ELU
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, Concatenate
from keras.optimizers import Adam
K.set_image_data_format("channels_last")


class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)


class ELU(ELU):
    def __init__(self, alpha=1.0, **kwargs):
        self.__name__ = "ELU"
        super(ELU, self).__init__(**kwargs)
        self.alpha = K.cast_to_floatx(alpha)


patch_size = 32


def tf_psnrmain(y_true, y_pred):
    return tf_psnr(y_true, y_pred, max_val=1.0, name=None)


def tf_psnr(a, b, max_val, name=None):
    with ops.name_scope(name, 'PSNR', [a, b]):
        max_val = math_ops.cast(max_val, a.dtype)

        mse = math_ops.reduce_mean(math_ops.squared_difference(a, b), [-4, -3, -2])
        psnr_val = math_ops.subtract(
            20 * math_ops.log(max_val) / math_ops.log(10.0),
            np.float64(10 / np.log(10)) * math_ops.log(mse),
            name='psnr')
        return array_ops.identity(psnr_val)


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def ssim(y_true, y_pred):
    return tf_ssim(y_true, y_pred, cs_map=False, mean_metric=True, size=11, sigma=1.5)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    img1 = tf.reshape(img1, [1, patch_size, -1, 1])
    img2 = tf.reshape(img2, [1, patch_size, -1, 1])

    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def custom_loss(y_true, y_pred):
    total_loss = mean_abs_error(y_true, y_pred) + grad_loss(y_true, y_pred)
    return total_loss


def mean_sq_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_abs_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def relu_advanced(x):
    return K.relu(x, max_value=1)


def grad_loss(y_true, y_pred):
    img1 = y_true
    img2 = y_pred
    patch_size = 32
    img1 = tf.reshape(img1, [1, patch_size, patch_size, -1, 1])
    img2 = tf.reshape(img2, [1, patch_size, patch_size, -1, 1])

    sx = tf.constant([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    sy = tf.transpose(sx)

    szx = [sx, sx, sx]
    szx = tf.expand_dims(szx, axis=-1)
    szx = tf.expand_dims(szx, axis=-1)

    szy = [sy, sy, sy]
    szy = tf.expand_dims(szy, axis=-1)
    szy = tf.expand_dims(szy, axis=-1)

    s_all1 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    s_all0 = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_all_neg1 = tf.constant([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    szz = [s_all1, s_all0, s_all_neg1]
    szz = tf.expand_dims(szz, axis=-1)
    szz = tf.expand_dims(szz, axis=-1)

    grad_x_true = tf.abs(tf.nn.conv3d(img1, szx, strides=[1, 1, 1, 1, 1], padding='VALID'))
    grad_y_true = tf.abs(tf.nn.conv3d(img1, szy, strides=[1, 1, 1, 1, 1], padding='VALID'))
    grad_z_true = tf.abs(tf.nn.conv3d(img1, szz, strides=[1, 1, 1, 1, 1], padding='VALID'))

    grad_x_pred = tf.abs(tf.nn.conv3d(img2, szx, strides=[1, 1, 1, 1, 1], padding='VALID'))
    grad_y_pred = tf.abs(tf.nn.conv3d(img2, szy, strides=[1, 1, 1, 1, 1], padding='VALID'))
    grad_z_pred = tf.abs(tf.nn.conv3d(img2, szz, strides=[1, 1, 1, 1, 1], padding='VALID'))

    grad_diff_x = tf.abs(grad_x_true - grad_x_pred)
    grad_diff_y = tf.abs(grad_y_true - grad_y_pred)
    grad_diff_z = tf.abs(grad_z_true - grad_z_pred)

    grad_mag = tf.reduce_sum(grad_diff_x + grad_diff_y + grad_diff_z)

    return grad_mag


def DenseNet(patch_size, growth_rate, no_layers, loss):
    n_channels = growth_rate
    input_shape = (patch_size[0], patch_size[1], patch_size[2], 1)

    inputs = Input(input_shape)
    # Initial Convolution Layer
    x = Conv3D(filters=2 * growth_rate, kernel_size=(3, 3, 3), padding='same', activation=ELU(alpha=1.0))(inputs)

    for i in range(no_layers):
        x_list = [x]
        cb = Conv3D(filters=2 * growth_rate, kernel_size=(3, 3, 3), padding='same', activation=ELU(alpha=1.0))(x)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)

        n_channels += growth_rate
        # for transititon layer
        x = Conv3D(n_channels, kernel_size=(1, 1, 1), padding='same', activation=ELU(alpha=1.0))(x)

    x = Conv3D(1, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)

    model = Model(inputs=inputs, outputs=x)

    adamOpt = Adam(lr=0.00001)
    model.compile(loss=loss, optimizer=adamOpt, metrics=[mean_sq_error, tf_psnrmain, ssim])
    model.summary(line_length=110)

    return model
