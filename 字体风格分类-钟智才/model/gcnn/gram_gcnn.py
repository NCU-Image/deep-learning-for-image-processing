import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation, Dropout,Input,ZeroPadding2D
'''
需要考虑采用BN和不采用BN的结果
'''
class GCNN(Model):

    def __init__(self):
        super(GCNN, self).__init__()

        self.c1 = tf.keras.models.Sequential([
            Conv2D(kernel_size=(4, 4), filters=64, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), strides=1, padding='same'),
            Dropout(0.2),
        ])

        self.c2 = tf.keras.models.Sequential([
            Conv2D(filters=128, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(0.2),
        ])

        self.c3 = tf.keras.models.Sequential([
            Conv2D(filters=256, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(0.2),
        ])

        self.c4 = tf.keras.models.Sequential([
            Conv2D(filters=512, kernel_size=(4, 4), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),
            Dropout(0.2),
        ])

        # 拉直层
        self.flatten = Flatten()  # 将数据拉直

        # gram矩阵操作
        self.gram_model = tf.keras.models.Sequential([
            ZeroPadding2D(padding=(1, 1)),
            Conv2D(kernel_size=19, filters=32, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(pool_size=(2, 2), padding='same'),  # 池层
            Dropout(0.2),
        ])

        # 全连接分类层
        self.full = tf.keras.models.Sequential([
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax'),
        ])

    # 计算均值
    def means(self, x):
        means = tf.reduce_mean(x, axis=3, keepdims=False)
        means = tf.reduce_mean(means, axis=2)
        means = tf.reduce_mean(means, axis=1)
        means = tf.expand_dims(means, -1)
        means = tf.expand_dims(means, -1)
        means = tf.expand_dims(means, -1)
        return means

    # 计算Gram矩阵
    def gram(self, x):
        gram_reslut = tf.linalg.einsum('bijc,bijd->bcd', x, x)
        gram_reslut = tf.expand_dims(gram_reslut, -1)  # 增加一维 还原(B,H,W,C)
        return gram_reslut


    def call(self, x):
        # 卷积层1
        conv_1 = self.c1(x)
        # 卷积层2
        conv_2 = self.c2(conv_1)
        # 卷积层3
        conv_3 = self.c3(conv_2)
        # 卷积层4
        conv_4 = self.c4(conv_3)

        # 计算gram矩阵
        gram_4 = self.gram(conv_4)

        # 求均值
        means_4 = self.means(gram_4)

        # 减均值
        gram = tf.subtract(gram_4, means_4)

        # 改变大小形状
        gram = tf.image.resize(gram, (224, 224))



        x = self.gram_model(gram)
        x = self.flatten(x)
        y =  self.full(x)

        return y

    def get_model(self):
        x = Input(shape=(64, 64, 1))
        return Model(inputs=[x], outputs=self.call(x))


if __name__ == '__main__':
    gcnn = GCNN()
    gcnn.get_model().summary()
