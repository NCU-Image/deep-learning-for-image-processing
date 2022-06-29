import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from  tensorflow.python.framework.ops import Tensor
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout


# 定义两个模型进行处理
# 一个模型处理四个图片数据
# 模型将前面模型的处理结果作为输入



# 局部卷积神经网络模型
from tensorflow.keras import Input
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Softmax


class TPCNN(Model):
    '''
        卷积核大小4*4 , 使用adam , 学习率10^-5,小批量大小80
        每个模型包括4个卷积层和两个完全连接层
        每次卷积操作之后，依次执行批标准化处理、ReLU非线性激活、最大池和退出。
        采用ReLU非线性激活，最大池
    '''

    def __init__(self):
        super(TPCNN,self).__init__()
        # 卷积层一
        self.c1 = Conv2D(filters=64, kernel_size=(4,4), padding='same')  # 卷积层
        self.b1 = BatchNormalization( )  # BN层
        self.a1 = Activation('relu')  # 激活函数层
        # version1.0
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')  # 池层
        #self.p1 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        # 卷积层二
        self.c2 = Conv2D(filters=128, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b2 = BatchNormalization( )  # BN层
        self.a2 = Activation('relu')  # 激活函数层
        # version1.0
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层
        # self.p2 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        # 卷积层三
        self.c3 = Conv2D(filters=256, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b3 = BatchNormalization( )  # BN层
        self.a3 = Activation('relu')  # 激活函数层
        # version1.0
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层
        # self.p3 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d3 = Dropout(0.2)  # dropout层

        # 卷积层四
        self.c4 = Conv2D(filters=512, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b4 = BatchNormalization( )  # BN层
        self.a4 = Activation('relu')  # 激活函数层
        #version1.0
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层
        # self.p4 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d4 = Dropout(0.2)  # dropout层

        # self.flatten = Flatten()
        # self.f1 = Dense(2048, activation='relu')
        # self.d5 = Dropout(0.5)
        # self.f2 = Dense(512, activation='relu')
        # self.d6 = Dropout(0.5)
        # self.f3 = Dense(4,activation='relu')


    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
       # x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
       # x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        #x = self.d3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p4(x)
        #x = self.d4(x)

        # x = self.flatten(x)
        # x = self.f1(x)
        # x = self.d5(x)
        # x = self.f2(x)
        # x = self.d6(x)
        # x = self.f3(x)

        return x
    def get_model(self):
        x = Input(shape=(32,32,1))
        return Model(inputs=[x],outputs=self.call(x))

def load_model(I1, I2, I3, I4, I5):
    tpcnn = TPCNN()

    I1 = Input(shape=I1[0].shape, name='in1')
    I2 = Input(shape=I2[0].shape, name='in2')
    I3 = Input(shape=I3[0].shape, name='in3')
    I4 = Input(shape=I4[0].shape, name='in4')
    I5 = Input(shape=I5[0].shape, name='in5')

    I1_out = tpcnn(I1)
    I2_out = tpcnn(I2)
    I3_out = tpcnn(I3)
    I4_out = tpcnn(I4)
    I5_out = tpcnn(I5)

    # 常见的融合操作

    # 1. 求和 ， 通过layers.Add(inputs）
    # 2， 求差 layers.Subtract(inputs)

    # test one
    # 3. 求平均
    outs = tf.keras.layers.Average()([I1_out, I2_out, I3_out, I4_out, I5_out])
    outs = Flatten()(outs)
    # 1. test 0.01 , activity_regularizer=tf.keras.regularizers.L1(0.000001)
    out = Dense(2048, activation='relu')(outs)
    #out = Dense(2048, activation=tf.nn.leaky_relu,activity_regularizer=tf.keras.regularizers.L1(0.000001))(outs)
    out = Dropout(0.5)(out)
    # , activity_regularizer=tf.keras.regularizers.L1(0.001)
    out = Dense(512, activation='relu')(out)
    #out = Dense(512, activation=tf.nn.leaky_relu,activity_regularizer=tf.keras.regularizers.L1(0.001))(out)
    out = Dropout(0.5)(out)
    out = Dense(4, activation='softmax')(out)

    model = Model(inputs=[I1, I2, I3, I4, I5], outputs=out)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    return model


def load_model2(I1, I2, I3, I4, I5):
    tpcnn = TPCNN()

    I1 = Input(shape=I1[0].shape, name='in1')
    I2 = Input(shape=I2[0].shape, name='in2')
    I3 = Input(shape=I3[0].shape, name='in3')
    I4 = Input(shape=I4[0].shape, name='in4')
    I5 = Input(shape=I5[0].shape, name='in5')

    I1_out = tpcnn(I1)
    I2_out = tpcnn(I2)
    I3_out = tpcnn(I3)
    I4_out = tpcnn(I4)
    I5_out = tpcnn(I5)

    # 常见的融合操作

    # 1. 求和 ， 通过layers.Add(inputs）
    # 2， 求差 layers.Subtract(inputs)

    # test one
    # 3. 求平均
    outs = tf.keras.layers.Average()([I1_out, I2_out, I3_out, I4_out, I5_out])
    outs = Flatten()(outs)
    # 1. test 0.01 , activity_regularizer=tf.keras.regularizers.L1(0.000001)
    out = Dense(2048, activation='relu')(outs)
    #out = Dense(2048, activation=tf.nn.leaky_relu,activity_regularizer=tf.keras.regularizers.L1(0.000001))(outs)
    out = Dropout(0.5)(out)
    # , activity_regularizer=tf.keras.regularizers.L1(0.001)
    out = Dense(512, activation='relu')(out)
    #out = Dense(512, activation=tf.nn.leaky_relu,activity_regularizer=tf.keras.regularizers.L1(0.001))(out)
    out = Dropout(0.5)(out)


    model = Model(inputs=[I1, I2, I3, I4, I5], outputs=out)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    return model



# Total params: 20,588,740
# Trainable params: 20,586,820
# Non-trainable params: 1,920
if __name__ == '__main__':
    x1 = Input(shape=(32, 32, 1))
    x2 = Input(shape=(32, 32, 1))
    x3 = Input(shape=(32, 32, 1))
    x4 = Input(shape=(32, 32, 1))
    x5 = Input(shape=(32, 32, 1))
    tpcnn = load_model(x1, x2, x3, x4,x5)
    tpcnn.summary()
    lcnn = TPCNN()
    # lcnn = load_model([x1],[x2],[x3],[x4])
    lcnn.get_model().summary()


