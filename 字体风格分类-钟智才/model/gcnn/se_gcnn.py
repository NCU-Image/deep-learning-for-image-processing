# 用于读取DATA数据文件，并生成训练集，验证集，测试集文件

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout
import os
# 全局卷积神经网络
from tensorflow.python.keras import Input
import numpy as np
import csv
import numpy as np

# 计算均值
def get_means(x):
    means = tf.reduce_mean(x, axis=3, keepdims=False)
    means = tf.reduce_mean(means, axis=2)
    means = tf.reduce_mean(means, axis=1)
    means = tf.expand_dims(means, -1)
    means = tf.expand_dims(means, -1)
    means = tf.expand_dims(means, -1)
    return means

class SEModel(Model):
    def __init__(self , in_chnnel):
        super(SEModel , self).__init__()
        # 1. 先通过Sequeze模块，将Feature map 进行压缩
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        # 2. 然后通过Bootleneck结构进行激活
        self.excitation = tf.keras.Sequential([
            Dense(in_chnnel / 16 , activation='relu'),
            Dense(in_chnnel , activation='sigmoid'),
        ])

    def call(self , inputs):

        squeeze = self.squeeze(inputs)

        excitation = self.excitation(squeeze)

        excitation = tf.keras.layers.Reshape((1 ,1 , inputs.shape[-1]))(excitation)

        scale = inputs * excitation # 将权重加到输入特征图上

        return scale


class GramGcnn(Model):
    def __init__(self):
        super(GramGcnn, self).__init__()

        self.conv1 = tf.keras.models.Sequential([
            Conv2D(kernel_size=(4, 4), filters=64, padding='same'),
            Activation(tf.nn.leaky_relu),
            MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])

        self.conv2 = tf.keras.models.Sequential([
            Conv2D(filters=128, kernel_size=(4, 4), padding='same'),
            Activation(tf.nn.leaky_relu),
            MaxPool2D(pool_size=(2, 2), padding='same'),
        ])

        self.conv3 = tf.keras.models.Sequential([
            Conv2D(filters=256, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
        ])

        self.conv4 = tf.keras.models.Sequential([
            Conv2D(filters=512, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
        ])

        self.se = SEModel(512)
        #采用GlobalAveragePooling2D还是全连接层视集体情况而定
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.outputs = Dense(4, activation='softmax')

    def call(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # 通过注意力机制
        x = self.se(conv4)

        x = self.gap(x)

        y = self.outputs(x)

        return y

    def get_model(self):
        x = Input(shape=(64, 64, 1))
        return Model(inputs=[x], outputs=self.call(x))


