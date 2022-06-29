from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
import os
from keras.optimizers import SGD
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D,BatchNormalization,Activation,MaxPool2D,Dropout
from keras.optimizers import RMSprop
from keras import backend as K, regularizers
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# from Dataset.Network.Resnet50 import Conv2d_BN,create_base_network
# from Dataset.Network.Vgg16 import conv_block,create_base_network
# from Dataset.Network.Xception import create_base_network
from Dataset.Network.Alex import Conv_block,create_base_network
num_classes = 5
batch_size=32
epochs = 100

# #欧氏距离
def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2
# def eucl_dist(inputs):
#     x, y = inputs
#     sum_square = K.sum(K.square(x-y),axis=1,keepdims=True)

    # return K.sqrt(K.maximum(sum_square,K.epsilon()))
# 损失函数
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# 创建样本对
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    labels_1 = []
    labels_2 = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    print(n)
    for d in range(num_classes):
    # 对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            # labels += [[d,d,1]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = random.randrange(1, num_classes)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % num_classes
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            labels_1 += [d,d]
            labels_2 += [d,dn]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels +=[1,0]
            # labels += [[d,dn,0]]
    # print("len of labels:", len(labels))
    # print("len of pairs:", len(pairs))
    # print("labels_1:",labels_1)
    # print("labels_2:",labels_2)
    # print("labels:",labels)
    # print(pairs)
    print("*****************")
    labels_1= to_categorical(labels_1, num_classes)
    labels_2 = to_categorical(labels_2, num_classes)
    labels = to_categorical(labels, 2)
    print("labels_1:", labels_1)
    print("labels_2:", labels_2)
    print("labels:", labels)
    return np.array(pairs),np.array(labels_1),np.array(labels_2), np.array(labels)
input_shape = (96, 96, 3)
img= os.listdir('E:\PycharmProjects\AI\Dataset\Fote Dataset\Train')
def read_image(img_name):
    im = Image.open(img_name).convert('RGB')
    data = np.array(im)
    return data
def read_lable(img_name):
    basename = os.path.basename(img_name) # 返回最后的路径
    data = basename.split('_')[0]
    return data
images = []
img_labels = []
for guy in os.listdir('E:\PycharmProjects\AI\Dataset\Fote Dataset\Train'):
    person_dir = os.path.join('E:\PycharmProjects\AI\Dataset\Fote Dataset\Train',guy)
    for i in os.listdir(person_dir):
        if i.endswith('.png'):
            fd = os.path.join(person_dir,i)
            images.append(read_image(fd))
            img_labels.append(read_lable(fd))
y = np.array(list(map(int,img_labels)))
x= np.array(images)
# x = x.resize(96,96)
# x= x.reshape(len(x),9216)
# print(x.size)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5,random_state=30)
print(y_test)
print(y_train)
# random_state = 20   7820  1750    782   175
# random_state = 30   7830  1830    783   183
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
im_weight = 96
im_height = 96
im_channel = 3
# print("x_train.shape", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], im_weight, im_height, im_channel)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], im_weight, im_height, im_channel)
# print("x_train.shape", x_train.shape)
# print("y_train.shape", y_train.shape)
# print("x_test.shape", x_test.shape)
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs,tr_y1,tr_y2, tr_y= create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs,te_y1,te_y2, te_y = create_pairs(x_test, digit_indices)


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
# network definition
base_network = create_base_network(input_shape)
input_a = Input(shape=input_shape,name='img_1')
input_b = Input(shape=input_shape,name='img_2')

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)
# let's add a fully-connected layer
category_predict1 = Dense(num_classes, activation='softmax', name='ctg_out_1')(
    Dropout(0.5)(processed_a)
)
category_predict2 = Dense(num_classes, activation='softmax', name='ctg_out_2')(
    Dropout(0.5)(processed_b)
)
distance = Lambda(eucl_dist,name='square')([processed_a, processed_b])
judge = Dense(2, activation='softmax', name='bin_out')(distance)
model = Model(inputs=[input_a, input_b], outputs=[category_predict1, category_predict2, judge])
model.summary()

model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss={'ctg_out_1': 'categorical_crossentropy',
                    'ctg_out_2': 'categorical_crossentropy',
                    'bin_out':  contrastive_loss
                        },
              loss_weights={
                      'ctg_out_1': 1,
                      'ctg_out_2': 1,
                      'bin_out': 0.5
                  },
              metrics=['accuracy'])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y1,tr_y2,tr_y],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], [te_y1,te_y2,te_y]),
          callbacks=[early_stopping, reduce_lr, save_model])

model.save('font_alex0.5.h5')

