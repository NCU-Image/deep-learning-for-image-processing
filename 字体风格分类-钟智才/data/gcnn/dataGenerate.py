# 用于读取DATA数据文件，并生成训练集，验证集，测试集文件
import json
import os
import tensorflow as tf
import numpy as np
from PIL import Image

"""
    生成GCNN训练数据。
    data_path： 设置为数据集路径
    data_path下面因包含Test，Train, Valid三个文件夹
    这里的数据集图片是64x64大小，如需要替换成96x96数据集，将64替换成96
"""
class DATA_GCNN():

    root_dir = '../data/gcnn/'

    x_train_savepath = root_dir  + 'x_train.npy'
    y_train_savepath = root_dir  + 'y_train.npy'

    x_test_savepath =  root_dir  + 'x_test.npy'
    y_test_savepath =  root_dir  + 'y_test.npy'

    x_valid_savepath =  root_dir  + 'x_valid.npy'
    y_valid_savepath = root_dir  + 'y_valid.npy'

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_valid = []
    y_valid = []
    data_path = r"E:\Pycharms\AI\Calligraphy\DATA"

    # 读取数据路径，并且判断是否存在数据文件，不存在就进行数据生成
    def __init__(self):
        # 如果存在生成的数据文件，就进行读取数据文件
        if os.path.exists(self.x_train_savepath) and os.path.exists(self.y_train_savepath) \
                and os.path.exists(self.x_test_savepath) and os.path.exists(self.y_test_savepath) \
                and os.path.exists(self.x_valid_savepath) and os.path.exists(self.y_valid_savepath):
            self.load_data()
        else:
            self.generate_data()
            self.load_data()

    def generate_data(self):
        print('-----------Generate Datasets--------')

        temp_x, temp_y = self.create_data(self.data_path + '/Test', "Test")
        self.x_test.extend(temp_x)
        self.y_test.extend(temp_y)
        temp_x, temp_y = self.create_data(self.data_path + '/Train', "Train")
        self.x_train.extend(temp_x)
        self.y_train.extend(temp_y)
        temp_x, temp_y = self.create_data(self.data_path + '/Valid', "Valid")
        self.x_valid.extend(temp_x)
        self.y_valid.extend(temp_y)

        print("--------Save Datasets-------------")
        # 打乱标签顺序
        # 打乱标签的顺序，这样就可以保证训练集和测试集不会被
        # 样本本身是有规律的，导致分配的验证集的标签可能在训练集中可能就没有。
        self.x_train = np.array(self.x_train, dtype=np.float32)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test, dtype=np.float32)
        self.y_test = np.array(self.y_test)
        self.x_valid = np.array(self.x_valid, dtype=np.float32)
        self.y_valid = np.array(self.y_valid)

        self.x_train, self.y_train = self.shuffe_data(self.x_train, self.y_train)
        self.x_test, self.y_test = self.shuffe_data(self.x_test, self.y_test)
        self.x_valid, self.y_valid = self.shuffe_data(self.x_valid, self.y_valid)

        self.x_train = np.reshape(self.x_train, (len(self.x_train), -1))
        self.x_valid = np.reshape(self.x_valid, (len(self.x_valid), -1))
        self.x_test = np.reshape(self.x_test, (len(self.x_test), -1))

        np.save(self.x_train_savepath, self.x_train)
        np.save(self.x_valid_savepath, self.x_valid)
        np.save(self.x_test_savepath, self.x_test)

        np.save(self.y_train_savepath, self.y_train)
        np.save(self.y_valid_savepath, self.y_valid)
        np.save(self.y_test_savepath, self.y_test)
        print("--------Save Datasets End-------------")


    def load_data(self):
        print('-----------Load Datasets------------')
        # 从文件读取数据集
        self.x_train = np.load(self.x_train_savepath)
        self.x_test = np.load(self.x_test_savepath)
        self.x_valid = np.load(self.x_valid_savepath)

        self.y_train = np.load(self.y_train_savepath)
        self.y_test = np.load(self.y_test_savepath)
        self.y_valid = np.load(self.y_valid_savepath)

        self.y_train = tf.convert_to_tensor(self.y_train)
        self.y_test = tf.convert_to_tensor(self.y_test)
        self.y_valid = tf.convert_to_tensor(self.y_valid)

        self.x_train = np.reshape(self.x_train, (len(self.x_train), 64, 64))
        self.x_test = np.reshape(self.x_test, (len(self.x_test), 64, 64))
        self.x_valid = np.reshape(self.x_valid, (len(self.x_valid), 64, 64))

        self.x_train = tf.convert_to_tensor(self.x_train)
        self.x_test = tf.convert_to_tensor(self.x_test)
        self.x_valid = tf.convert_to_tensor(self.x_valid)

        self.x_train = tf.expand_dims(self.x_train, -1)
        self.x_test = tf.expand_dims(self.x_test, -1)
        self.x_valid = tf.expand_dims(self.x_valid, -1)
        print('----------Load Datasets End---------')


    def load_train(self):
        return self.x_train, self.y_train


    def load_valid(self):
        return self.x_valid, self.y_valid


    def load_test(self):
        return self.x_test, self.y_test



    def shuffe_data(self, x_train, y_train):
        index = [i for i in range(len(x_train))]
        np.random.shuffle(index)
        temp_y = []
        temp_x = []
        for i in index:
            temp_y.append(y_train[i])
            temp_x.append(x_train[i])
        return temp_x, temp_y


    # 读取文件的数据
    def create_data(self, path, ftlag):
        x_train, y_train = [], []
        dict = {}
        try:
            j = 0
            for file in os.listdir(path=path):
                tag = file  # 获取标签
                tag_file = path + "/" + file
                img_list = os.listdir(tag_file)  # 进入下一级目录
                img_list.sort()
                tag_len = len(img_list)
                # 读取训练集
                for i in range(0, tag_len):
                    img_path = tag_file + "/" + img_list[i]
                    img = Image.open(img_path)
                    img = np.array(img.convert('L'))
                    img = img / 255.0  # 归一化处理
                    x_train.append(img)
                    y_train.append(j)
                dict[j] = tag
                j = j + 1
            with open(ftlag + 'mapLabel.txt', 'w') as f:
                f.write(json.dumps(dict))
            return x_train, y_train
            # x_train = np.array(x_train,dtype=np.float32)  # 将数据转化为一维数组格式
            # y_train = np.array(y_train)  # 将数据转化为一维数组格式
            # return x_train, y_train
        except Exception as e:
            print(e)

if __name__ == '__main__':
    test = DATA_GCNN()