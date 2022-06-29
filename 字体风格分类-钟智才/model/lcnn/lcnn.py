import tensorflow as tf
from keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout,Average


# 定义两个模型进行处理
# 一个模型处理四个图片数据
# 模型将前面模型的处理结果作为输
# 局部卷积神经网络模型

class LCNN(Model):
    '''
        卷积核大小4*4 , 使用adam , 学习率10^-5,小批量大小80
        每个模型包括4个卷积层和两个完全连接层
        每次卷积操作之后，依次执行批标准化处理、ReLU非线性激活、最大池和退出。
        采用ReLU非线性激活，最大池
    '''

    def __init__(self):
        super(LCNN,self).__init__()
        # 卷积层一
        self.c1 = Conv2D(filters=64, kernel_size=(4,4), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活函数层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same')  # 池层

        # 卷积层二
        self.c2 = Conv2D(filters=128, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('relu')  # 激活函数层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层

        # 卷积层三
        self.c3 = Conv2D(filters=256, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('relu')  # 激活函数层
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层

        # 卷积层四
        self.c4 = Conv2D(filters=512, kernel_size=(4, 4), padding='same')  # 卷积层
        self.b4 = BatchNormalization()  # BN层
        self.a4 = Activation('relu')  # 激活函数层
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池层


    def call(self, x):

        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)


        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p4(x)


        return x

    def get_model(self):
        x = Input(shape=(32,32,1))
        return Model(inputs=[x],outputs=self.call(x))

def load_model(I1, I2, I3, I4):
    flcnn = LCNN()

    I1 = Input(shape=I1[0].shape, name='in1')
    I2 = Input(shape=I2[0].shape, name='in2')
    I3 = Input(shape=I3[0].shape, name='in3')
    I4 = Input(shape=I4[0].shape, name='in4')

    I1_out = flcnn(I1)
    I2_out = flcnn(I2)
    I3_out = flcnn(I3)
    I4_out = flcnn(I4)

    # test one
    # 3. 求平均
    outs = Average()([I1_out, I2_out, I3_out, I4_out])

    outs = Flatten()(outs)
    out = Dense(2048, activation='relu')(outs)
    out = Dropout(0.5)(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(4, activation='softmax')(out)

    model = Model(inputs=[I1, I2, I3, I4], outputs=out)
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
    lcnn = load_model(x1,x2,x3,x4)
    lcnn.summary()
    lcnn = LCNN()
   # lcnn = load_model([x1],[x2],[x3],[x4])
    lcnn.get_model().summary()

