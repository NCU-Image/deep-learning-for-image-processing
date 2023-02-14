import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.train_utils import (get_lr_scheduler, set_optimizer_lr, weights_init)
import nets
from utils.callback import LossHistory
from utils.dataloader import dataset_collate, TestDataSet
from utils.paper_utils import get_num_classes, show_config
from utils.utils_fit_normal import fit_one_epoch

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    annotation_path = "cls_train.txt"
    annotation_path_val = "cls_val.txt"
    # annotation_path_test = "cls_test.txt"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    # input_shape     = [160, 160, 3]
    input_shape = [160, 160, 3]
    # --------------------------------------------------------#
    #   主干特征提取网络的选择
    # res50
    # gcnn
    # --------------------------------------------------------#
    backbone = "gcnn"
    # model_path      = "model_data/facenet_mobilenet.pth"
    model_path = ""
    pretrained = False

    # batch_size      = 288
    batch_size = 12
    Init_Epoch = 0
    Epoch = 100
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 50
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'
    # ------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0  
    # ------------------------------------------------------------------#
    num_workers = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0

    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    model = nets.GcnnCbam(backbone=backbone, num_classes=num_classes)
    # model = nets.GCNN()

    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    model_train = model.train()

    # -------------------------------------------------------#
    #   0.2用于验证，0.8用于训练
    # -------------------------------------------------------#
    val_split = 0.2
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    with open(annotation_path_val, "r") as f:
        lines_val = f.readlines()
    # with open(annotation_path_test, "r") as f:
    #     lines_test = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    num_train = len(lines)
    num_val = int(len(lines_val))
    # num_test = int(len(lines_test))

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        # epoch_step_test = num_test // batch_size
        # ---------------------------------------#

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        train_dataset = TestDataSet(input_shape, lines, num_classes, random=True)
        val_dataset = TestDataSet(input_shape, lines_val, num_classes, random=True)
        # test_dataset = TestDataSet(input_shape, lines_test, num_classes, random=True)
        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)
        #
        # gen_test = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size // 3, num_workers=num_workers,
        #                       pin_memory=True,
        #                       drop_last=True)

        if Cuda:
            model.cuda()

        for epoch in range(Init_Epoch, Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val
                          , gen, gen_val, Epoch, Cuda, save_period, save_dir)

        loss_history.writer.close()
