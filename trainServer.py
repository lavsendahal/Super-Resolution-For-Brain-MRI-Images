import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, Activation, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import os

patch_size = 32
K.set_image_data_format("channels_last")

from keras.layers.advanced_activations import LeakyReLU, ELU
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))


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


def mean_sq_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


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


def relu_advanced(x):
    return K.relu(x, max_value=1)


def DenseNet(patch_size=32, growth_rate=24, no_layers=8):
    n_channels = growth_rate
    input_shape = (patch_size, patch_size, patch_size, 1)

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
    model.compile(loss='mean_absolute_error', optimizer=adamOpt, metrics=[mean_sq_error, tf_psnrmain, ssim])
    model.summary(line_length=110)
    return model


if __name__ == '__main__':
    class ELU(ELU):
        def __init__(self, alpha=1.0, **kwargs):
            self.__name__ = "ELU"
            super(ELU, self).__init__(**kwargs)
            self.alpha = K.cast_to_floatx(alpha)


    class LeakyReLU(LeakyReLU):
        def __init__(self, **kwargs):
            self.__name__ = "LeakyReLU"
            super(LeakyReLU, self).__init__(**kwargs)


    train_batch_size = 8
    reduceLearningRate = 0.5

    print('-' * 60)
    print('Loading and preprocessing train data 32x32x32 Patch Size..')
    print('-' * 60)
    trainImg = np.load('patches_3d/gauss_sigma_0_5/train_img_lr32.npy')
    trainGt = np.load('patches_3d/gauss_sigma_0_5/train_img_hr32.npy')

    print('-' * 60)
    print('Loading and preprocessing validation data 32x32x32 Patch Size..')
    print('-' * 60)
    valImg = np.load('patches_3d/gauss_sigma_0_5/val_img_lr32.npy')
    valGt = np.load('patches_3d/gauss_sigma_0_5/val_img_lr32.npy')

    print('-' * 60)
    print('Creating and compiling model..')
    print('-' * 60)
    # create a model
    model = DenseNet(patch_size=32, growth_rate=8, no_layers=8)

    print('-' * 60)
    print('Fitting model...')
    print('-' * 60)

    # ============================================================================
    print('training starting..')

    if 'outputs_gauss_sigma_1' not in os.listdir(os.curdir):
        os.mkdir('outputs_gauss_sigma_1')

    log_filename = 'outputs_gauss_sigma_1/' + '3dPatch' + '_model_train.csv'

    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)

    checkpoint_filepath = 'outputs_gauss_sigma_1/' + 'model-{epoch:03d}.h5'

    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')

    callbacks_list = [csv_log, checkpoint]
    callbacks_list.append(ReduceLROnPlateau(factor=reduceLearningRate, patience=30,
                                            verbose=True))
    callbacks_list.append(EarlyStopping(verbose=True, patience=20))

    # ============================================================================
    history = model.fit(trainImg, trainGt, epochs=500, verbose=1, batch_size=train_batch_size,
                        validation_data=(valImg, valGt), shuffle=True, callbacks=callbacks_list)

    model_name = 'outputs_gauss_sigma_1/' + '3dPatch32' + '_model_last'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'
