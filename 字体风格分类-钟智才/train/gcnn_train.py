import math
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger

from tensorflow.keras import Input
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.keras.layers import concatenate, Dropout
from matplotlib import pyplot as plt
import os


import tensorflow as tf

from data.gcnn.dataGenerate import DATA_GCNN
from model.gcnn.gcnn import GCNN
from utils.data_util import get_init_epoch
from utils.utils import step_decay


def first_gcnn_train(model_name,index,batch_size=None):

    # 定义参数
    Epochs = 180 # 训练轮数
    init_Epoch = 0 # 起始训练轮次 ， 主要用于断点续训
    if  not batch_size:
        batch_size = 16
    if not os.path.exists('./'+model_name):
        os.mkdir(model_name)

    path_head = './' + model_name + '/' + index
    if not  os.path.exists(path_head):
        os.mkdir(path_head)

    # 文件路径 checkpoint/
    log_path = path_head + '/training_log'
    # log_path = './training_log'+'/'+dirname #训练数据保存位置
    checkpoint_save_path = path_head + '/checkpoint/Baseline.ckpt'
    #checkpoint_save_path = './checkpoint/'+dirname+'/Baseline.ckpt' # 模型权重保存路径
    checkpoint_save_best_path = path_head + '/best_checkpoint/Baseline.ckpt'
    #checkpoint_save_best_path = './best_checkpoint/'+dirname+'/Baseline.ckpt' # 最优模型保存路径
    logdir = path_head +  '/logs' # 日志保存路径

    if os.path.exists(log_path):
        init_Epoch = get_init_epoch(log_path)



    data_set = DATA_GCNN()
    x_train, y_train = data_set.load_train()
    x_valid, y_valid = data_set.load_valid()

    y_train = y_train.numpy()
    y_valid = y_valid.numpy()
    y_train = np.concatenate([y_train, y_valid], axis=0)
    y_train = tf.convert_to_tensor(y_train)

    x_train = x_train.numpy()
    x_valid = x_valid.numpy()
    x_train = np.concatenate([x_train, x_valid], axis=0)
    x_train = tf.convert_to_tensor(x_train)


    x_valid, y_valid = data_set.load_test()
    # 加载模型
    model = GCNN()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    #加载模型权重
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------load the model------')
        model.load_weights(checkpoint_save_path)

    # tensorboard 可视化数据
    log_dir = logdir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #定义回调函数
    cp_callback_save = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True) #用于保存权重数据
    # 通过设置model可以决定最优模型是loss还是acc model='min'表示loss model-'max'表示acc
    cp_callback_save_best = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_best_path,
                                                               monitor='val_sparse_categorical_accuracy',
                                                               save_weights_only=True, save_best_only=True,
                                                               mode='max')  # 用于保存最优权重


    csv_logger = CSVLogger(log_path,append=True)

    # 定义学习率调整器
    learningSchudel = LearningRateScheduler(step_decay)

    history = model.fit(x_train, y_train, epochs=Epochs,initial_epoch=init_Epoch, batch_size=batch_size,validation_data=(x_valid,y_valid),validation_freq=1, callbacks=[csv_logger,learningSchudel,cp_callback_save,cp_callback_save_best])



if __name__ == '__main__':

    first_gcnn_train('time_1','1',batch_size=8)# 73s  229ms epoch , 64s 201 ms 53s 166 ms 54ms 169ms