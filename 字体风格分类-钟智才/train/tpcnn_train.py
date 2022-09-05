import math
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import concatenate, Dropout

from matplotlib import pyplot as plt
import os

from data.tpcnn.dataGenerate import DATA_TPCNN
from model.tpcnn.tpcnn import load_model
from utils.data_util import get_init_epoch
from utils.utils import step_decay


def first_tpcnn_train(model_name, index):
    # 定义参数
    Epochs = 180  # 训练轮数
    init_Epoch = 0  # 起始训练轮次 ， 主要用于断点续训
    batch_size = 8
    if not os.path.exists('./' + model_name):
        os.mkdir(model_name)

    path_head = './' + model_name + '/' + index

    if not os.path.exists(path_head):
        os.mkdir(path_head)
    # 文件路径 checkpoint/
    log_path = path_head + '//training_log'
    # log_path = './training_log'+'/'+dirname #训练数据保存位置
    checkpoint_save_path = path_head + '/checkpoint/Baseline.ckpt'
    # checkpoint_save_path = './checkpoint/'+dirname+'/Baseline.ckpt' # 模型权重保存路径
    checkpoint_save_best_path = path_head + '/best_checkpoint/Baseline.ckpt'
    # checkpoint_save_best_path = './best_checkpoint/'+dirname+'/Baseline.ckpt' # 最优模型保存路径
    logdir = path_head + '/logs'  # 日志保存路径

    if os.path.exists(log_path):
        init_Epoch = get_init_epoch(log_path)

    # 从数据集读取数据加载器
    data_loader = DATA_TPCNN()

    # 加载训练集和验证集
    x_I1_train, x_I2_train, x_I3_train, x_I4_train, x_I5_train, y_train = data_loader.load_train()
    x_I1_valid, x_I2_valid, x_I3_valid, x_I4_valid, x_I5_valid, y_valid = data_loader.load_valid()

    y_train = y_train.numpy()
    y_valid = y_valid.numpy()
    y_train = np.concatenate([y_train, y_valid], axis=0)
    y_train = tf.convert_to_tensor(y_train)

    x_I1_train = x_I1_train.numpy()
    x_I1_valid = x_I1_valid.numpy()
    x_I1_train = np.concatenate([x_I1_train, x_I1_valid], axis=0)
    x_I1_train = tf.convert_to_tensor(x_I1_train)

    x_I2_train = x_I2_train.numpy()
    x_I2_valid = x_I2_valid.numpy()
    x_I2_train = np.concatenate([x_I2_train, x_I2_valid], axis=0)
    x_I2_train = tf.convert_to_tensor(x_I2_train)

    x_I3_train = x_I3_train.numpy()
    x_I3_valid = x_I3_valid.numpy()
    x_I3_train = np.concatenate([x_I3_train, x_I3_valid], axis=0)
    x_I3_train = tf.convert_to_tensor(x_I3_train)

    x_I4_train = x_I4_train.numpy()
    x_I4_valid = x_I4_valid.numpy()
    x_I4_train = np.concatenate([x_I4_train, x_I4_valid], axis=0)
    x_I4_train = tf.convert_to_tensor(x_I4_train)

    x_I5_train = x_I5_train.numpy()
    x_I5_valid = x_I5_valid.numpy()
    x_I5_train = np.concatenate([x_I5_train, x_I5_valid], axis=0)
    x_I5_train = tf.convert_to_tensor(x_I5_train)

    # 用测试集测试模型
    x_I1_valid, x_I2_valid, x_I3_valid, x_I4_valid, x_I5_valid, y_valid = data_loader.load_test()

    # 加载模型
    model = load_model(x_I1_train, x_I2_train, x_I3_train, x_I4_train, x_I5_train)

    # 加载模型权重
    if os.path.exists(checkpoint_save_path + '.index'):
        print('------load the model------')
        model.load_weights(checkpoint_save_path)

    # tensorboard 可视化数据

    # 定义回调函数
    cp_callback_save = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                          save_weights_only=True)  # 用于保存权重数据
    cp_callback_save_best = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_best_path,
                                                               monitor='val_sparse_categorical_accuracy',
                                                               save_weights_only=True, save_best_only=True,
                                                               mode='max')  # 用于保存最优权重
    csv_logger = CSVLogger(log_path, append=True)

    # 定义学习率调整器
    learningSchudel = LearningRateScheduler(step_decay)

    history = model.fit([x_I1_train, x_I2_train, x_I3_train, x_I4_train, x_I5_train], y_train, epochs=Epochs,
                        initial_epoch=init_Epoch, batch_size=batch_size,
                        validation_data=([x_I1_valid, x_I2_valid, x_I3_valid, x_I4_valid, x_I5_valid], y_valid),
                        validation_freq=1, callbacks=[csv_logger, cp_callback_save, cp_callback_save_best])


if __name__ == '__main__':
    first_tpcnn_train('time', '1')
