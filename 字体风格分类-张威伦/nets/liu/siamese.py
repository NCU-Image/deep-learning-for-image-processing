import keras.backend as K
from keras.layers import Input, Dense
from keras.layers import Lambda
from keras.models import Model

import nets


# 孪生神经网络
def siamese(input_shape):
    vgg_model = nets.VGG16()

    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    # 我们将两个输入传入到主干特征提取网络
    encoded_image_1 = vgg_model.call(input_image_1)
    encoded_image_2 = vgg_model.call(input_image_2)

    # 相减取绝对值
    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    # 两次全连接
    out = Dense(512, activation='relu')(l1_distance)
    # 值，固定在0-1
    out = Dense(1, activation='sigmoid')(out)

    model = Model([input_image_1, input_image_2], out)
    return model