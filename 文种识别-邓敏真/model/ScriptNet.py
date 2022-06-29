import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D,LSTM,Lambda,add,concatenate,Activation,DepthwiseConv2D,LeakyReLU
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D,BatchNormalization,Reshape,AveragePooling1D,AveragePooling2D,Dropout
from keras import backend as K

# def CNN(num_classes,img_w,img_h,channels):
#     # 224，224，3
#
# def mobilenet_block(x,filters,strides):
#     x = DepthwiseConv2D(kernel_size=3,strides=strides,padding='same')(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#
#     x = Conv2D(filters=filters,kernel_size=1,strides=1)(x)
#     x = BatchNormalization()(x)
#     x = LeakyReLU()(x)
#     return x
def CNN():
    INPUT_SHAPE = 224,224,3
    input = Input(shape=(INPUT_SHAPE))

    # 第一个卷积部分
    # 112，112，64
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(input)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)

    # 第二个卷积部分
    # 56,56,128
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    # 第三个卷积部分
    # 28,28,256
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    # 第四个卷积部分
    # 14,14,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    # 第五个卷积部分
    # 7,7,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)
    # # 维度变换
    inner = Reshape(target_shape=((7, 3584)), name='reshape')(x)  # (None, 56, 7168)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 56, 64)

    # RNN layer
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 56, 512)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 56, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)

    lstm_2 = LSTM(256, return_sequences=False, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=False, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)
    reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 56, 1024)
    lstm2_merged = BatchNormalization()(lstm2_merged)


  ######################################################################################################################
    y = Conv2D(64, (3, 3), activation='relu', padding='same', name='blocky1_conv1')(input)
    y = Conv2D(64, (3, 3), activation='relu', padding='same', name='blocky1_conv2')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='blocky1_pool')(y)

    z1 = GlobalAveragePooling2D()(y)

    # # 第二个卷积部分
    # # 56,56,128
    y = Conv2D(128, (3, 3), activation='relu', padding='same', name='blocky2_conv1')(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same', name='blocky2_conv2')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='blocky2_pool')(y)

    z2 = GlobalAveragePooling2D()(y)

    # 第三个卷积部分
    # 28,28,256
    y = Conv2D(256, (3, 3), activation='relu', padding='same', name='blocky3_conv1')(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', name='blocky3_conv2')(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', name='blocky3_conv3')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='blocky3_pool')(y)

    z3 = GlobalAveragePooling2D()(y)

    # 第四个卷积部分
    # 14,14,512
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky4_conv1')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky4_conv2')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky4_conv3')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='blocky4_pool')(y)

    z4 = GlobalAveragePooling2D()(y)
    print(z4)
    # 第五个卷积部分
    # 7,7,512
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky5_conv1')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky5_conv2')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='blocky5_conv3')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='blocky5_pool')(y)

    z5 = GlobalAveragePooling2D()(y)
    print(z5)
    z = concatenate([z4,z5])

    f = concatenate([lstm2_merged,z])

    inner = Dense(1024, kernel_initializer='he_normal', name='dense2')(f)
    inner = Dense(4, kernel_initializer='he_normal', name='dense2')(inner)  # (None, 4)
    inner = Dropout(0.2)(inner)
    y_pred = Activation('softmax', name='softmax')(inner)
    return Model(inputs=[input], outputs=y_pred)

model = CNN()
model.summary()