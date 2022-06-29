import tensorflow as tf
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout
import os
# 全局卷积神经网络
from tensorflow.python.keras import Input


class GCNN(Model):
    '''
        卷积核大小4*4 , 使用adam , 学习率10^-5,小批量大小80
        每个模型包括4个卷积层和两个完全连接层
        每次卷积操作之后，依次执行批标准化处理、ReLU非线性激活、最大池和退出。
        采用ReLU非线性激活，最大池
    '''
    def __init__(self):
        super(GCNN,self).__init__()
        # 卷积层一
        self.c1 = Conv2D(kernel_size=(4,4),filters=64,padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活函数层
        self.p1 = MaxPool2D(pool_size=(2, 2),strides=1, padding='same')  # 池层
        self.d1 = Dropout(0.2)  # dropout层

        # 卷积层二
        self.c2 = Conv2D(filters=128, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('relu')  # 激活函数层
        # version1.0
        self.p2 = MaxPool2D(pool_size=(2, 2), padding='same')  # 池层
        # self.p2 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        # 卷积层三
        self.c3 = Conv2D(filters=256, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活函数层
        # version1.0
        self.p3 = MaxPool2D(pool_size=(2, 2), padding='same')  # 池层
        # self.p3 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d3 = Dropout(0.2)  # dropout层

        # 卷积层四
        self.c4 = Conv2D(filters=512, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b4 = BatchNormalization()  # BN层
        self.a4 = Activation('relu')  # 激活函数层
        # version1.0
        self.p4 = MaxPool2D(pool_size=(2, 2), padding='same')  # 池层
        # self.p4 = MaxPool2D(pool_size=(1,1),strides=1,padding='same')
        self.d4 = Dropout(0.2)  # dropout层

        # 神经网络
        self.flatten = Flatten()  # 将数据拉直
        self.f1 = Dense(2048, activation='relu' )  # 全连接层
        self.d5 = Dropout(0.2)  # dropout层
        self.f2 = Dense(512, activation='relu' )  # 全连接层
        self.d6 = Dropout(0.2)  # dropout层
        self.f3 = Dense(4, activation='softmax')  # 全连接层

    def call(self,x):

        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.flatten(x)

        x = self.f1(x)
        x = self.d5(x)
        x = self.f2(x)
        x = self.d6(x)
        y = self.f3(x)

        return y

    def get_model(self):
        x = Input(shape=(64,64,1))
        return Model(inputs=[x],outputs=self.call(x))

#70,918,468

if __name__ == '__main__':
    # 用于获取计算图
    # log_dir = './log'
    # writter = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))

    gcnn = GCNN()
    gcnn.get_model().summary()
    #
    #
    # # 开启计算图追踪
    # tf.summary.trace_on(True, )
    #
    # # 导出计算图
    # with writter.as_default():
    #     tf.summary.trace_export('encoder', step=0, profiler_outdir='./log')