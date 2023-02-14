import os

import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

# 十进制转换二进制
def decToBin(num):
    arry = []  # 定义一个空数组，用于存放2整除后的商
    while True:
        arry.append(str(num % 2))  # 用列表的append方法追加
        num = num // 2  # 用地板除求num的值
        if num == 0:  # 若地板除后的值为0，那么退出循环
            break

    return "".join(arry[::-1])  # 列表切片倒叙排列后再用join拼接


# 计算 旋转不变性的最小值
def minBinary(pixel):
    length = len(pixel)
    if length == 0:
        print("ok")
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '00000000'


lis = []
for i in range(0, 256):
    minb = decToBin(i)
    # 转换成2进制
    val = minBinary(minb)
    v = int(val, 2)
    if v not in lis:
        lis.append(v)


def getLBPH(img_lbp, numPatterns, grid_x, grid_y, normed):
    img_lbp = np.reshape(img_lbp, (64, -1))
    '''
    计算LBP特征图像的直方图LBPH
    '''
    h, w = img_lbp.shape
    width = int(w / grid_x)
    height = int(h / grid_y)
    # 定义LBPH的行和列，grid_x*grid_y表示将图像分割的块数，numPatterns表示LBP值的模式种类
    result = np.zeros((grid_x * grid_y, numPatterns), dtype=float)
    resultRowIndex = 0
    # 对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
    for i in range(grid_x):
        for j in range(grid_y):
            # 图像分块
            src_cell = img_lbp[i * height:(i + 1) * height, j * width:(j + 1) * width]
            # 计算直方图
            hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), normed)
            # 将直方图放到result中
            result[resultRowIndex] = hist_cell
            resultRowIndex += 1
    return np.reshape(result, (-1))


def getLocalRegionLBPH(src, minValue, maxValue, normed):
    '''
    计算一个LBP特征图像块的直方图
    '''
    data = np.reshape(src, (-1))
    # 计算得到直方图bin的数目，直方图数组的大小
    bins = maxValue - minValue + 1;
    # 定义直方图每一维的bin的变化范围
    ranges = (float(minValue), float(maxValue + 1))
    hist, bin_edges = np.histogram(src, bins=bins, range=ranges, normed=False)

    if normed:
        hist = hist / sum(hist)
    return hist


# clf = svm.SVC(probability=True)
def SVM_f(data_train_x=0, data_train_y=0, data_val_x=0, data_val_y=0, data_test_x=0, data_test_y=0, C=1.0):
    clf = svm.SVC(C=C, kernel='rbf', probability=True)
    clf.fit(data_train_x, data_train_y)
    pre_train = clf.predict(data_train_x)
    pre_val = clf.predict(data_val_x)
    print("trian:", accuracy_score(data_train_y, pre_train))
    print("val:", accuracy_score(data_val_y, pre_val))

    pre_test = clf.predict(data_test_x)
    print("test:", accuracy_score(data_test_y, pre_test))


def C_lbp(data_0, data_1, data_2, data_3, C=1.0, lis=64 * 256, normed=False):
    data_train_x, data_train_y, data_val_x, data_val_y, data_test_x, data_test_y = get_data(data_0, data_1, data_2,
                                                                                            data_3, lis=lis,
                                                                                            normed=normed)
    # print(data_train_x.shape)
    retu = SVM_f(data_train_x, data_train_y, data_val_x, data_val_y, data_test_x, data_test_y, C=C)
    return retu


def get_data(data_0, data_1, data_2, data_3, normed=False, lis=64 * 256):
    np.random.shuffle(data_0)
    np.random.shuffle(data_1)
    np.random.shuffle(data_2)
    np.random.shuffle(data_3)

    data_0 = data_0[:1800]
    data_1 = data_1[:1800]
    data_2 = data_2[:1800]
    data_3 = data_3[:1800]

    normed = normed
    # 对风格求 LBPH，整幅图求 LBPH
    # liu
    liu_feature_256_1 = np.zeros((data_0.shape[0], lis))
    for i in range(0, data_0.shape[0]):
        if lis == 256:
            hist, bin_edges = np.histogram(data_0[i, 1:data_0.shape[1] - 1], bins=lis, range=(0, lis), normed=normed)
        else:
            hist = getLBPH(data_0[i, 1:data_0.shape[1] - 1], 256, 8, 8, normed)
        liu_feature_256_1[i] = hist
    liu_feature_256_1 = np.c_[liu_feature_256_1, np.zeros((data_0.shape[0]))]

    # ou
    ou_feature_256_1 = np.zeros((data_1.shape[0], lis))
    for i in range(0, data_1.shape[0]):
        if lis == 256:
            hist, bin_edges = np.histogram(data_1[i, 1:data_1.shape[1] - 1], bins=lis, range=(0, lis), normed=normed)
        else:
            hist = getLBPH(data_1[i, 1:data_1.shape[1] - 1], 256, 8, 8, normed)
        ou_feature_256_1[i] = hist
    ou_feature_256_1 = np.c_[ou_feature_256_1, np.ones((data_1.shape[0]))]

    # yan
    yan_feature_256_1 = np.zeros((data_2.shape[0], lis))
    for i in range(0, data_2.shape[0]):
        if lis == 256:
            hist, bin_edges = np.histogram(data_2[i, 1:data_2.shape[1] - 1], bins=lis, range=(0, lis), normed=normed)
        else:
            hist = getLBPH(data_2[i, 1:data_2.shape[1] - 1], 256, 8, 8, normed)
        yan_feature_256_1[i] = hist
    yan_feature_256_1 = np.c_[yan_feature_256_1, np.ones((data_2.shape[0])) * 2]

    # zhao
    zhao_feature_256_1 = np.zeros((data_3.shape[0], lis))
    for i in range(0, data_3.shape[0]):
        if lis == 256:
            hist, bin_edges = np.histogram(data_3[i, 1:data_3.shape[1] - 1], bins=lis, range=(0, lis), normed=normed)
        else:
            hist = getLBPH(data_3[i, 1:data_3.shape[1] - 1], 256, 8, 8, normed)
        zhao_feature_256_1[i] = hist
    zhao_feature_256_1 = np.c_[zhao_feature_256_1, np.ones((data_3.shape[0])) * 3]

    liu_train, liu_val, liu_test = liu_feature_256_1[:540], liu_feature_256_1[540:720], liu_feature_256_1[720:]
    ou_train, ou_val, ou_test = ou_feature_256_1[:540], ou_feature_256_1[540:720], ou_feature_256_1[720:]
    yan_train, yan_val, yan_test = yan_feature_256_1[:540], yan_feature_256_1[540:720], yan_feature_256_1[720:]
    zhao_train, zhao_val, zhao_test = zhao_feature_256_1[:540], zhao_feature_256_1[540:720], zhao_feature_256_1[
                                                                                                720:]

    data_train = np.concatenate((liu_train, ou_train, yan_train, zhao_train), axis=0)
    data_val = np.concatenate((liu_val, ou_val, yan_val, zhao_val), axis=0)
    data_test = np.concatenate((liu_test, ou_test, yan_test, zhao_test), axis=0)

    np.random.shuffle(data_train)
    np.random.shuffle(data_val)
    np.random.shuffle(data_test)

    data_train_x, data_train_y = data_train[:, :-1], data_train[:, -1]
    data_val_x, data_val_y = data_val[:, :-1], data_val[:, -1]
    data_test_x, data_test_y = data_test[:, :-1], data_test[:, -1]
    print(data_test_x.shape, data_test_y.shape)
    #     with open('.\\LBP_feature_data.json', 'a+') as f:
    #         json.dump(data_test_x.tolist(), f)
    #     with open('.\\LBP_feature_label.json', 'a+') as f:
    #         json.dump(data_test_y.tolist(), f)
    return data_train_x, data_train_y, data_val_x, data_val_y, data_test_x, data_test_y


def to_csv(gen_path: str, data_liu, data_ou, data_yan, data_zhao):
    ret = pd.DataFrame(data_liu)
    ret.to_csv(os.path.join(gen_path, "/0.csv"))

    ret = pd.DataFrame(data_ou)
    ret.to_csv(os.path.join(gen_path, "/1.csv"))

    ret = pd.DataFrame(data_yan)
    ret.to_csv(os.path.join(gen_path, "/2.csv"))

    ret = pd.DataFrame(data_zhao)
    ret.to_csv(os.path.join(gen_path, "/3.csv"))


def img_resize(data_style_liu, data_style_ou, data_style_yan, data_style_zhao,
               gen_path: str):
    """
    输入路径
    :param gen_path:
    :param data_style_liu:
    :param data_style_ou:
    :param data_style_yan:
    :param data_style_zhao:
    :return:
    """
    data_liu = 0
    data_ou = 0
    data_yan = 0
    data_zhao = 0

    for path in data_style_liu:
        print(path)
        if 'DS_Store' in path:
            continue
        image = cv2.imread(os.path.join(gen_path, "/gen/0/" + path), 0)
        image = np.reshape(image, (-1))
        image = np.append(image, 0)
        image = np.array([image])
        # print(image.shape)
        if type(data_liu) == int:
            data_liu = image
        else:
            data_liu = np.concatenate((data_liu, image), axis=0)

    # ou
    for path in data_style_ou:
        print(path)
        if 'DS_Store' in path:
            continue
        image = cv2.imread(os.path.join(gen_path, "/gen/1/" + path), 0)
        image = np.reshape(image, (-1))
        image = np.append(image, 1)
        image = np.array([image])
        if type(data_ou) == int:
            data_ou = image
        else:
            data_ou = np.concatenate((data_ou, image), axis=0)

    # yan
    for path in data_style_yan:
        print(path)
        if 'DS_Store' in path:
            continue
        image = cv2.imread(os.path.join(gen_path, "/gen/2/" + path), 0)
        image = np.reshape(image, (-1))
        image = np.append(image, 2)
        image = np.array([image])
        if type(data_yan) == int:
            data_yan = image
        else:
            data_yan = np.concatenate((data_yan, image), axis=0)

    # zhao
    for path in data_style_zhao:
        print(path)
        if 'DS_Store' in path:
            continue
        image = cv2.imread(os.path.join(gen_path, "/gen/3/" + path), 0)
        image = np.reshape(image, (-1))
        image = np.append(image, 3)
        image = np.array([image])
        if type(data_zhao) == int:
            data_zhao = image
        else:
            data_zhao = np.concatenate((data_zhao, image), axis=0)


def min_binary(pixel):
    length = len(pixel)
    if length == 0:
        print("ok")
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '00000000'


def origin_lbp(img, min_binary_bool=False):
    frame = np.zeros(img.shape, dtype=img.dtype) * 255
    # print(image)
    img = np.array(img, dtype=np.int32)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            value = img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1]
            value += img[i][j - 1] + img[i][j] + img[i][j + 1]
            value += img[i + 1][j - 1] + img[i + 1][j] + img[i + 1][j + 1]
            value /= 9
            v_1 = str(int(img[i - 1][j - 1] > value))
            v_2 = str(int(img[i - 1][j] > value))
            v_3 = str(int(img[i - 1][j + 1] > value))
            v_4 = str(int(img[i][j + 1] > value))
            v_5 = str(int(img[i + 1][j + 1] > value))
            v_6 = str(int(img[i + 1][j] > value))
            v_7 = str(int(img[i + 1][j - 1] > value))
            v_8 = str(int(img[i][j - 1] > value))
            binary_value = v_1 + v_2 + v_3 + v_4 + v_5 + v_6 + v_7 + v_8

            if min_binary_bool:
                binary_value = min_binary(str(binary_value))
            frame[i][j] = int(binary_value, 2)

    return frame


min_Bin = False


def c_hog():
    one = r'/Users/william/Documents/数据集/lishu_resize/邓石如'
    two = r'/Users/william/Documents/数据集/lishu_resize/伊秉绶'
    three = r'/Users/william/Documents/数据集/lishu_resize/金农'
    forth = r'/Users/william/Documents/数据集/lishu_resize/郑簠'
    gen_path = r'/Users/william/Documents/数据集/lishu_resize/gen/'
    # data_style_liu = os.listdir(one)
    # data_style_ou = os.listdir(two)
    # data_style_yan = os.listdir(three)
    # data_style_zhao = os.listdir(forth)
    # for i in data_style_liu:
    #     print(i)
    #     if 'DS_Store' in i:
    #         continue
    #     image = cv2.imread(os.path.join(one, "/" + str(i)), 0)
    #     img = origin_lbp(image, min_Bin)
    #     cv2.imwrite(os.path.join(one, "/gen/0" + str(i)), img)
    #
    # for i in data_style_ou:
    #     print(i)
    #     if 'DS_Store' in i:
    #         continue
    #     image = cv2.imread(os.path.join(two, "/" + str(i)), 0)
    #     img = origin_lbp(image, min_Bin)
    #     cv2.imwrite(os.path.join(one, "/gen/1" + str(i)), img)
    # for i in data_style_yan:
    #     print(i)
    #     if 'DS_Store' in i:
    #         continue
    #     image = cv2.imread(os.path.join(three, "/" + str(i)), 0)
    #     img = origin_lbp(image, min_Bin)
    #     cv2.imwrite(os.path.join(one, "/gen/2" + str(i)), img)
    # for i in data_style_zhao:
    #     print(i)
    #     if 'DS_Store' in i:
    #         continue
    #     image = cv2.imread(os.path.join(forth, "/" + str(i)), 0)
    #     img = origin_lbp(image, min_Bin)
    #     cv2.imwrite(os.path.join(one, "/gen/3" + str(i)), img)
    # style 0-liu 1-ou 2-yan 3-zhao
    data_0 = np.array(pd.read_csv(os.path.join(gen_path, "0.csv")), dtype='float32')
    data_1 = np.array(pd.read_csv(os.path.join(gen_path, "1.csv")), dtype='float32')
    data_2 = np.array(pd.read_csv(os.path.join(gen_path, "2.csv")), dtype='float32')
    data_3 = np.array(pd.read_csv(os.path.join(gen_path, "3.csv")), dtype='float32')
    C_lbp(data_0, data_1, data_2, data_3)


if __name__ == '__main__':
    c_hog()