from __future__ import print_function
from __future__ import absolute_import
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import keras
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import Conv2D
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D,MaxPool2D,Dropout,ZeroPadding2D
from keras import layers
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras import Input
from keras import backend as K
from keras.applications import Xception
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,TensorBoard
from keras.layers import Dense, Dropout, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import plot_model

# set_session(tf.Session(config=config))
from keras.utils import plot_model
classnum = 4
train_datagen = ImageDataGenerator(

    rescale=1. / 255,
    # shear_range=20.0,  # 错切变换，浮点数，剪切强度
    # rotation_range=10,  # 旋转变换，整数，随机旋转的度数范围
    # horizontal_flip=True,
    # vertical_flip=True\
    # width_shift_range=0.5,
# channel_shift_range=10

        )

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
train_generator = train_datagen.flow_from_directory(
        'F:\Siamese_Keras\Dataset\single_gusi_data\Train',
        target_size=(96, 96),
        # batch_size=1,
        batch_size=batch_size,
        class_mode='categorical',
        )

validation_generator = test_datagen.flow_from_directory(
        'F:\Siamese_Keras\Dataset\single_gusi_data\Valid',
        target_size=(96, 96),
        # batch_size=1,
        batch_size=batch_size,
        class_mode='categorical')



# 网络结构 VGG16
def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def create_base_network(input_shape):
    '''
    Base network to be shared.
    '''
    input = Input(shape=input_shape)
    # stage 1
    x = conv_block(input, filters=64, kernel_size=(3, 3), name="conv1_1_64_3x3_1")
    x = conv_block(x, filters=64, kernel_size=(3, 3), name="conv1_2_64_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_1_2x2_2")(x)
    # stage 2
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_1_128_3x3_1")
    x = conv_block(x, filters=128, kernel_size=(3, 3), name="conv2_2_128_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_2_2x2_2")(x)
    # stage 3
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_1_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(3, 3), name="conv3_2_256_3x3_1")
    x = conv_block(x, filters=256, kernel_size=(1, 1), name="conv3_3_256_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_3_2x2_2")(x)
    # stage 4
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv4_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv4_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_4_2x2_2")(x)
    # stage 5
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_1_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(3, 3), name="conv5_2_512_3x3_1")
    x = conv_block(x, filters=512, kernel_size=(1, 1), name="conv5_3_512_3x3_1")
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool_5_2x2_2")(x)

    # FC layers
    # FC layer 1
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # FC layer 2
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    # # FC layer 3
    x = Dense(classnum)(x)
    x = BatchNormalization()(x)
    x = Activation("softmax")(x)
    return Model(input,x)
log_dir= 'log'
save_model = ModelCheckpoint(
                                        log_dir + 'style_sia_vgg16_1{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='acc',
                                        save_weights_only=False,
                                        save_best_only=True,
                                        period=10
                                    )
# 学习率下降的方式，acc三次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
                                monitor='acc',
                                factor=0.5,
                                patience=3,
                                verbose=1
                            )
# 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=1
                            )

if os.path.exists('gusi_vgg1.h5'):
    model = load_model('gusi_vgg1.h5')
else:
    # create the base pre-trained model
    input_shape = (96, 96, 3)
    base_network = create_base_network(input_shape)
    input_tensor = Input(shape=input_shape,name='img')
    predict = base_network(input_tensor)

    # feature1 = feature(img1)
    # feature2 = feature(img2)
    # # let's add a fully-connected layer
    # category_predict1 = Dense(5, activation='softmax', name='ctg_out_1')(
    #     Dropout(0.5)(feature1)
    # )
    # category_predict2 = Dense(5, activation='softmax', name='ctg_out_2')(
    #     Dropout(0.5)(feature2)
    # )

    # concatenated = keras.layers.concatenate([feature1, feature2])
    # dis = Lambda(eucl_dist, name='square')([feature1, feature2])
    # concatenated = Dropout(0.5)(concatenated)
    # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(concatenated)

    # judge = Dense(2, activation='softmax', name='bin_out')(dis)
    model = Model(inputs=input_tensor, outputs=[predict])
    model.summary()
    # model.save('dog_xception_0.h5')
    # plot_model(model, to_file='../logs/model_combined.png')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='nadam',
    #               loss={'ctg_out_1': 'categorical_crossentropy',
    #                     'ctg_out_2': 'categorical_crossentropy',
    #                     'bin_out': 'categorical_crossentropy'},
    #               loss_weights={
    #                   'ctg_out_1': 1.,
    #                   'ctg_out_2': 1.,
    #                   'bin_out': 1.
    #               },
    #               metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    # model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
    #                     steps_per_epoch=16500/batch_size+1,
    #                     epochs=50,
    #                     validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
    #                     validation_steps=1800/batch_size+1,
    #                     callbacks=[early_stopping, auto_lr, save_model,tensorboard])
    # model.save('font_xception.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# train_generator = test_datagen.flow_from_directory(
#         'E:\PycharmProjects\AI\Dataset_Font\Train',
#         target_size=(299, 299),
#         # batch_size=1,
#         batch_size=batch_size,
#         class_mode='categorical')

# let's visualize layer names and layer indices to see how many layers
# we should freeze:

# for i, layer in enumerate(model.layers):
#     print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:

# cur_base_model = model.layers[2]
# for layer in cur_base_model.layers[:105]:
#     layer.trainable = False
# for layer in cur_base_model.layers[105:]:
#     layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
batch_size =32
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# save_model = ModelCheckpoint(log_dir + 'xception-tuned-{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
model.fit_generator(train_generator,
                    steps_per_epoch=2488/batch_size+1,
                    epochs=100,
                    validation_data=validation_generator,
                    validation_steps=532/batch_size+1,
                    callbacks=[early_stopping, reduce_lr, save_model]) # otherwise the generator would loop indefinitely
model.save('gusi_vgg1.h5')