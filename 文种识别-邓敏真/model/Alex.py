from __future__ import print_function
from __future__ import absolute_import
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D,BatchNormalization,Activation,MaxPool2D,Dropout
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D

#网络结构Alexnet
def Conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding="valid", name=None):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    num_classes=4
    x = Conv_block(input, filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", name="Conv_1_96_11x11_4")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(x)
    x = Conv_block(x, filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", name="Conv_2_256_5x5_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_3x3_2")(x)
    x = Conv_block(x, filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", name="Conv_3_384_3x3_1")
    x = Conv_block(x, filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", name="Conv_4_384_3x3_1")
    x = Conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", name="Conv_5_256_3x3_1")
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(x)

    x = Flatten()(x)
    x = Dense(units=4096)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=4096)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Dense(units=num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)
    return Model(input, x)