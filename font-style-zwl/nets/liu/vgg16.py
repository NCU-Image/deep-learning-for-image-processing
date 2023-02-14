from __future__ import print_function
from __future__ import absolute_import
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Activation, MaxPool2D, \
    Dropout
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D


# 网络结构 VGG16
def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def create_base_network(input_shape):
    '''
    Base network to be shared.
    '''
    input = Input(shape=input_shape)
    # stage 1
    x = conv_block(input, filters=64, kernel_size=(3, 3), name="conv1_1_64_3x3_1")
    x = conv_block(x, filters=64, kernel_size=(3, 3), name="conv1_2_64_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1_2x2_2")(x)
    # stage 2
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_1_128_3x3_1")
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_2_128_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2_2x2_2")(x)
    # stage 3
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_1_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_2_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(1, 1), name="conv3_3_256_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3_2x2_2")(x)
    # stage 4
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv4_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4_2x2_2")(x)
    # stage 5
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv5_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5_2x2_2")(x)

    # FC layers
    # FC layer 1
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # FC layer 2
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # # FC layer 3
    # x = Dense(num_classes)(x)
    # x = BatchNormalization()(x)
    # x = Activation("softmax")(x)
    return Model(input, x)
