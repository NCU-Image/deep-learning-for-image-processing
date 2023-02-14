import os
from shutil import copy, rmtree
import random

import cv2
import numpy as np
from PIL import Image


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_dataset():
    data_root = r"/Users/william/Documents/数据集/Llishu6"
    # origin_flower_path = os.path.join(data_root, "flower_photos")
    origin_flower_path = r"/Users/william/Documents/数据集/lishu"
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]
    train_root = os.path.join(data_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))
    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        # 随机采样验证集的索引
        for index, image_path in enumerate(images):
            if 'DS_Store' in image_path:
                continue
            image = cvtColor(Image.open(os.path.join(cla_path, image_path)))
            print(image.size)
            image = resize_image(image, [64, 64], letterbox_image=True)
            print(image.size)
            resize_cla = os.path.join('/Users/william/Documents/数据集/lishu_resize', cla)
            resize_img_path = os.path.join(resize_cla, image_path)
            image.save(resize_img_path)


def parse_data(k: int):
    # 保证随机可复现 10份交叉验证
    random.seed(k)

    # 3:1
    # 将数据集中10%的数据划分到验证集中
    split_rate_val = 0.25

    # 指向你解压后的flower_photos文件夹
    # cwd = os.getcwd()
    # data_root = os.path.join(cwd, "dataset")
    data_root = f"C:\project\lishu"
    # origin_flower_path = os.path.join(data_root, "flower_photos")
    origin_flower_path = "C:\project\lishu"
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "Train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "Valid")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num * split_rate_val))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
        for index, image in enumerate(images):
            if image not in eval_index:
                # 将分配至测试集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)

            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()


def parse_cross_data(k: int):
    # 保证随机可复现 10份交叉验证
    random.seed(k)

    # 8 ： 1 ： 1
    # 将数据集中10%的数据划分到验证集中
    split_rate_val = 0.2
    # 将数据集中10%的数据划分到验证集中
    split_rate_test = 0.2

    # 指向你解压后的flower_photos文件夹
    # cwd = os.getcwd()
    # data_root = os.path.join(cwd, "dataset")
    data_root = f"C:\project\lishu_cross\cross{k}"
    # origin_flower_path = os.path.join(data_root, "flower_photos")
    origin_flower_path = "C:\project\lishu"
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "Train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "Valid")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 建立保存test集的文件夹
    test_root = os.path.join(data_root, "Test")
    mk_file(test_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(test_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num * split_rate_val))
        test_index = random.sample(list(set(images) - set(eval_index)), k=int(num * split_rate_test))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
        for index, image in enumerate(images):
            if image in test_index:
                # 将分配至测试集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
        for index, image in enumerate(images):
            if image not in test_index and image not in eval_index:
                # 将分配至测试集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)

            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()


print("processing done!")

if __name__ == '__main__':
    # parse_data(1)
    resize_dataset()
    # for i in range(10):
    #     parse_cross_data(i)
