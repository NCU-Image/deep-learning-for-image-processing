import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, Dense, Activation, Dropout,Input

# 定义两个模型进行处理
# 一个模型处理四个图片数据
# 模型将前面模型的处理结果作为输入
class TPCNN(Model):
    '''
        卷积核大小4*4 , 使用adam , 学习率10^-5,小批量大小80
        每个模型包括4个卷积层和两个完全连接层
        每次卷积操作之后，依次执行批标准化处理、ReLU非线性激活、最大池和退出。
        采用ReLU非线性激活，最大池
    '''

    def __init__(self):
        super(TPCNN,self).__init__()

        self.c1 = tf.keras.models.Sequential([
            Conv2D(kernel_size=(4, 4), filters=64, padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])

        self.c2 = tf.keras.models.Sequential([
            Conv2D(filters=128, kernel_size=(4, 4), padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), strides=1, padding='same'),
        ])

        self.c3 = tf.keras.models.Sequential([
            Conv2D(filters=256, kernel_size=(4, 4), padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
        ])

        self.c4 = tf.keras.models.Sequential([
            Conv2D(filters=512, kernel_size=(4, 4), padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
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
        x = Input(shape=(32,32,1))
        return Model(inputs=[x],outputs=self.call(x))

# 均值处理
def fn_means(x):
    means = tf.reduce_mean(x, axis=3, keepdims=False)
    means = tf.reduce_mean(means, axis=2)
    means = tf.reduce_mean(means, axis=1)
    means = tf.expand_dims(means, -1)
    means = tf.expand_dims(means, -1)
    means = tf.expand_dims(means, -1)
    return means



# 计算Gram矩阵
def fn_gram(x):

    gram_reslut = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    gram_reslut = tf.expand_dims(gram_reslut, -1)  # 增加一维 还原(B,H,W,C)
    return gram_reslut




def load_model(I1, I2, I3, I4,I5):
    tpcnn = TPCNN()

    I1 = Input(shape=I1[0].shape, name='in1')
    I2 = Input(shape=I2[0].shape, name='in2')
    I3 = Input(shape=I3[0].shape, name='in3')
    I4 = Input(shape=I4[0].shape, name='in4')
    I5 = Input(shape=I5[0].shape, name='in5')

    full = tf.keras.models.Sequential([
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax'),
    ])

    gram_model =  tf.keras.models.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
            Conv2D(kernel_size=19, filters=32, padding='same'),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),  # 池层
            Dropout(0.2)
    ])

    I1_out = tpcnn(I1)
    I2_out = tpcnn(I2)
    I3_out = tpcnn(I3)
    I4_out = tpcnn(I4)
    I5_out = tpcnn(I5)

    outs = tf.keras.layers.Average()([I1_out, I2_out, I3_out, I4_out, I5_out])

    # 计算gram矩阵
    gram_avg =  fn_gram(outs)

    # 求均值
    means_avg = fn_means(gram_avg)

    # 减均值
    gram = tf.subtract(gram_avg, means_avg)

    # 改变大小形状
    gram = tf.image.resize(gram, (224, 224))

    x = gram_model(gram)


    out = full(x)

    model = Model(inputs=[I1, I2, I3, I4, I5], outputs=out)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    return model



