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
    INPUT_SHAPE = 32,128,3
    input = Input(shape=(INPUT_SHAPE))
    # x = Conv2D(filters=32,kernel_size=3,strides=2,padding='same')(input)
    # x = BatchNormalization()(x)
    # x = LeakyReLU()(x)
    #
    # x = mobilenet_block(x,filters=64,strides=1)
    #
    # x = mobilenet_block(x, filters=128, strides=2)
    # x = mobilenet_block(x, filters=128, strides=1)
    #
    # x = mobilenet_block(x, filters=256, strides=2)
    # x = mobilenet_block(x, filters=256, strides=1)
    #
    # x = mobilenet_block(x, filters=512, strides=2)
    #
    # for _ in range(5):
    #     x = mobilenet_block(x, filters=512, strides=1)
    #
    # x = mobilenet_block(x, filters=1024, strides=2)
    #
    # x = mobilenet_block(x, filters=1024, strides=1)
    #
    # x = AveragePooling2D(pool_size=7,strides=1)(x)
    #
    # x = Reshape(target_shape=(1,1024),name='reshape')(x)
    # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(x)  # (None, 4, 256)
    # lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(x)
    # reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
    #
    # lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None,4, 256)
    # lstm1_merged = AveragePooling1D(pool_size=1, strides=None)(lstm1_merged)
    # lstm1_merged = BatchNormalization()(lstm1_merged)
    #
    # lstm_2 = LSTM(256, return_sequences=False, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    # lstm_2b = LSTM(256, return_sequences=False, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
    #     lstm1_merged)
    # reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)
    #
    # lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 512)
    # lstm2_merged = BatchNormalization()(lstm2_merged)
    # lstm2_merged = Reshape(target_shape=(1,512))(lstm2_merged)

    # y = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(input)
    # y = BatchNormalization()(y)
    # y = LeakyReLU()(y)
    #
    # y = mobilenet_block(y, filters=64, strides=1)
    #
    # y = mobilenet_block(y, filters=128, strides=2)
    # y = mobilenet_block(y, filters=128, strides=1)
    #
    # y = mobilenet_block(y, filters=256, strides=2)
    # y = mobilenet_block(y, filters=256, strides=1)
    #
    # y = mobilenet_block(y, filters=512, strides=2)
    #
    # for _ in range(5):
    #     y = mobilenet_block(y, filters=512, strides=1)
    #
    # y = mobilenet_block(y, filters=1024, strides=2)
    # y3 = Reshape(target_shape=(1, 50176))(y)
    # y = mobilenet_block(y, filters=1024, strides=1)
    # y1 = Reshape(target_shape=(1,50176))(y)
    # y = AveragePooling2D(pool_size=7, strides=1)(y)
    # y2 = Reshape(target_shape=(1,1024))(y)
    # y = concatenate([y1,y2,y3])
    #
    # lstm2_merged = Flatten()(y)
    #
    # # lstm2_merged = concatenate([lstm2_merged,y])
    #
    # # lstm2_merged = Flatten()(lstm2_merged)
    # output = Dense(units=4, activation='softmax')(lstm2_merged)  # (None, 4)
    #
    # return Model(inputs=input,outputs=output)


    # input_shape = (img_w, img_h, channels)
    # inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    # #
    # # input= (224,224,3)
    # # model = InceptionV3(input_shape=input,weights='imagenet', include_top=False)
    # #
    # # z = model.output
    # # print(z)
    #
    #
    #
    # 第一个卷积部分
    # 112，112，64

    # # 分类部分
    # # 7x7x512
    # # 25088
    # ##############################################################################################
    # # y = Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu')(inputs)
    # # y = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')(y)
    # # y = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu')(y)
    # # y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same')(y)
    # # y = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # # y = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # # y = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same')(y)

    # inner_1 = Flatten()(z)
    # #
    #
    # ###############################################################################################



    # split = Reshape(target_shape=(7,3584),name='reshape')(x)
    # # y = concatenate([x,y])
    # # split = Reshape(target_shape=(7,5376),name='reshape')(y)
    #
    #
    # split = Reshape(target_shape=(7,3584),name='reshape')(x)
    # # split = Reshape(target_shape=(4,512),name='reshape')(x)
    # lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(split)  # (None, 4, 256)
    # lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(split)
    # reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
    # #
    # lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None,4, 256)
    # lstm1_merged = AveragePooling1D(pool_size=1, strides=None)(lstm1_merged)
    # lstm1_merged = BatchNormalization()(lstm1_merged)
    #
    # lstm_2 = LSTM(256, return_sequences=False, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    # lstm_2b = LSTM(256, return_sequences=False, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
    #     lstm1_merged)
    # reversed_lstm_2b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_2b)
    # lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 512)
    # lstm2_merged = BatchNormalization()(lstm2_merged)
    # lstm2_merged = Reshape(target_shape=(4,128),name='reshape12')(lstm2_merged)
    #
    # lstm2_merged = concatenate([lstm2_merged,inner_1])
    #
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv_1')(input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='Maxpooling_1')(x)


    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='Maxpooling_2')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv_3')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv_4')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='Maxpooling_3')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv_5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='Maxpooling_4')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv_6')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='Maxpooling_5')(x)
    print(x)
    inner = Reshape(target_shape=((4,2048)), name='reshape')(x)  # (None, 56, 7168)
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


    # y = Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu')(inputs)
    # y = MaxPooling2D(pool_size=(3,3),strides=(2,2))(y)
    # y = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu')(y)
    # y = MaxPooling2D(pool_size=(3,3),strides=(2,2))(y)
    # y = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # y = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # y = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    # y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(y)
    # y = Reshape(target_shape=(1,6400),name='reshape465')(y)
    # y = Reshape(target_shape=(4,1600),name="reshapej")(y)
    # split = concatenate([lstm2_merged, y])

    # split = Flatten()(lstm2_merged)

    inner = Dense(1024, kernel_initializer='he_normal', name='dense2')(lstm2_merged)
    inner = Dense(4, kernel_initializer='he_normal', name='dense2')(inner)  # (None, 4)
    inner = Dropout(0.5)(inner)
    y_pred = Activation('softmax', name='softmax')(inner)

    return Model(inputs=[input], outputs=y_pred)
# model = CNN(4,224,224,3)
model = CNN()
# model.summary()