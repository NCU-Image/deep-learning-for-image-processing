import shutil
import os

from keras.backend import set_session
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

# from Siamese_Font.Xception.font_train import pair_generator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import operator

from os import remove, path

from os.path import exists

from A_keras_dog.util import fwrite

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

train_datagen = ImageDataGenerator(
        rescale=1./255,
     )


batch_size = 32
train_generator = train_datagen.flow_from_directory(
        'F:\Siamese_Keras\Dataset\single_gusi_data\Train',
        target_size=(96, 96),
        # batch_size=1,
        batch_size=batch_size,
        class_mode='categorical')


# model = load_model('xception/xception-tuned-cont09-0.82.h5')
model = load_model('gusi_vgg1.h5')
# model = load_model('xception/xception-tuned-cont-froze-03-0.84.h5')
# single_model = Model(inputs=model.layers[0].input, outputs=[model.layers[6].output])
# single_model = Model(inputs=(model.layers[0].input,model.layers[1].input), outputs=[model.layers[8].output])
# model = single_model
# plot_model(model, to_file='single_model.png')
test_datagen = ImageDataGenerator(rescale=1./255,)
valid_generator = test_datagen.flow_from_directory(
    'F:\Siamese_Keras\Dataset\single_gusi_data\Valid',
    target_size=(96, 96),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

test_path = 'F:\Siamese_Keras\Dataset\single_gusi_data\Test'
# sorted() 函数对所有可迭代的对象进行排序操作。
# reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
print(label_idxs[0])
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(96, 96),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
# print test_generator.filenameenames

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
loss_1,acc = model.evaluate_generator(test_generator)
print(loss_1,acc)
#分类测试
# y = model.predict_generator(test_generator,workers=0, use_multiprocessing=True)

#0-1测试
# y = model.predict_generator(pair_generator(test_generator,batch_size=batch_size),workers=0,steps=5000//batch_size)
# num =10
# for i in range(num):
#     print("y_[i]:", y[i])
# y_max_idx = np.argmax(y, 1)  #取最大值时的参数（索引）
# print("y_max_idx",y_max_idx)

# y_max_es = np.max(y, 1)  # 值最大
# predict_path = 'font_predict.txt'
# if path.exists(predict_path):
#     remove(predict_path)
#
# fo = open(predict_path,"w")
#
# for i, idx in enumerate(y_max_idx):
#     fo.write(label_idxs[idx][0] + '\t' + test_generator.filenames[i][:-4])
#
# loss,acc = model.evaluate_generator(test_generator,workers=0, use_multiprocessing=True)
# print(loss,acc)

# new_test_path = '/home/cwh/coding/data/cwh/test_p'
# if not os.path.exists(new_test_path):
#     os.makedirs(new_test_path)


# for i, idx in enumerate(y_max_idx):
#     fwrite(predict_path , np.array(label_idxs)[idx][0] + '\t' + np.array(test_generator.filenames)[i][:-4] + '\n')
    # if y_max_es[i] > 0.9:
    #     if not os.path.exists(new_test_path + '/' + str(label_idxs[idx][0])):
    #         os.makedirs(new_test_path + '/' + str(label_idxs[idx][0]))
    #     shutil.copy(test_path + '/' + test_generator.filenames[i],
    #                 new_test_path + '/' + str(label_idxs[idx][0]) + '/' + test_generator.filenames[i][2:])