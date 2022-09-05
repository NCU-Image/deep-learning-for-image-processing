import os

import numpy as np

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout, Input



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

        # scale = inputs * excitation # 将权重加到输入特征图上

        return excitation
class TPCNN(Model):
    '''
        卷积核大小4*4 , 使用adam , 学习率10^-5,小批量大小80
        每个模型包括4个卷积层和两个完全连接层
        每次卷积操作之后，依次执行批标准化处理、ReLU非线性激活、最大池和退出。
        采用ReLU非线性激活，最大池
    '''

    def __init__(self):
        super(TPCNN, self).__init__()

        self.c1 = tf.keras.models.Sequential([
            Conv2D(kernel_size=(4, 4), filters=64, padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(1, 1), strides=1, padding='same'),
            Dropout(0.2) ,
        ])

        self.c2 = tf.keras.models.Sequential([
            Conv2D(filters=128, kernel_size=(4, 4), padding='same'),
            Activation('relu'),  # 激活函数层
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(0.2),
        ])

        self.c3 = tf.keras.models.Sequential([
            Conv2D(filters=256, kernel_size=(4, 4), padding='same'),  # 卷积层
            BatchNormalization(),  # BN层
            Activation('relu'),  # 激活函数层
            MaxPool2D(pool_size=(2, 2), padding='same'),  # 池层
            Dropout(0.2),
        ])

        self.c4 = tf.keras.models.Sequential([
            Conv2D(filters=512, kernel_size=(4, 4), padding='same'),  # 卷积层
            BatchNormalization(),  # BN层
            Activation('relu'),  # 激活函数层
            MaxPool2D(pool_size=(2, 2), padding='same'),  # 池层
            Dropout(0.2),
        ])


    def call(self, x):
        # 卷积层1
        conv_1 = self.c1(x)
        # 卷积层2
        conv_2 = self.c2(conv_1)
        # 卷积层3
        conv_3 = self.c3(conv_2)
        # 卷积层4
        conv_4 = self.c4(conv_3)

        return conv_4

    def get_model(self):
        x = Input(shape=(32, 32, 1))
        return Model(inputs=[x], outputs=self.call(x))


def load_model(I1, I2, I3, I4, I5):
    tpcnn = TPCNN()

    I1 = Input(shape=I1[0].shape, name='in1')
    I2 = Input(shape=I2[0].shape, name='in2')
    I3 = Input(shape=I3[0].shape, name='in3')
    I4 = Input(shape=I4[0].shape, name='in4')
    I5 = Input(shape=I5[0].shape, name='in5')



    gap = tf.keras.layers.GlobalAveragePooling2D()
    outputs = Dense(4 , activation='softmax')

    # 得到各个神经网络的输出结果
    I1_out = tpcnn(I1)
    I2_out = tpcnn(I2)
    I3_out = tpcnn(I3)
    I4_out = tpcnn(I4)
    I5_out = tpcnn(I5)

    I1_SE = SEModel(512)
    I2_SE = SEModel(512)
    I3_SE = SEModel(512)
    I4_SE = SEModel(512)
    I5_SE = SEModel(512)


    # 计算各个通道的权重信息
    I1_q = I1_SE(I1_out)
    I2_q= I2_SE(I2_out)
    I3_q = I3_SE(I3_out)
    I4_q = I4_SE(I4_out)
    I5_q = I5_SE(I5_out)

    F_1 = I1_out * I1_q
    F_2 = I2_out * I2_q
    F_3 = I3_out * I3_q
    F_4 = I4_out * I4_q
    F_5 = I5_out * I5_q

    F_m = tf.keras.layers.Add()([F_1,F_2,F_3,F_4,F_5])
    Q_m = tf.keras.layers.Add()([I1_q,I2_q,I3_q,I4_q,I5_q])
    F_m = tf.divide(F_m , Q_m) # 得到F_m的表示

    #用全连接还是GlobalAveragePooling2D视具体情况而定
    full = tf.keras.models.Sequential([
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax'),
    ])


    outs = gap(F_m)

    out = outputs(outs)

    model = Model(inputs=[I1, I2, I3, I4, I5], outputs=out)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    return model

