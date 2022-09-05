import json

import numpy as np
from PIL import Image
import os
import tensorflow as tf

"""
    生成TPCNN训练数据。
    data_path： 设置为数据集路径
    data_path下面因包含Test，Train, Valid三个文件夹
    这里的数据集图片是64x64大小，如需要替换成96x96数据集，将64替换成96，将32替换成48
"""
class DATA_TPCNN():
    root_dir = '../data/tpcnn/'
    x_trainI1_savepath = root_dir + 'x_trainI1.npy'
    x_trainI2_savepath = root_dir + 'x_trainI2.npy'
    x_trainI3_savepath = root_dir + 'x_trainI3.npy'
    x_trainI4_savepath = root_dir + 'x_trainI4.npy'
    x_trainI5_savepath = root_dir + 'x_trainI5.npy'
    y_train_savepath = root_dir + 'y_train.npy'

    x_validI1_savepath = root_dir + 'x_validI1.npy'
    x_validI2_savepath = root_dir + 'x_validI2.npy'
    x_validI3_savepath = root_dir + 'x_validI3.npy'
    x_validI4_savepath = root_dir + 'x_validI4.npy'
    x_validI5_savepath = root_dir + 'x_validI5.npy'
    y_valid_savepath = root_dir + 'y_valid.npy'

    x_testI1_savepath = root_dir + 'x_testI1.npy'
    x_testI2_savepath = root_dir + 'x_testI2.npy'
    x_testI3_savepath = root_dir + 'x_testI3.npy'
    x_testI4_savepath = root_dir + 'x_testI4.npy'
    x_testI5_savepath = root_dir + 'x_testI5.npy'
    y_test_savepath = root_dir + 'y_test.npy'

    # data_path = r"E:\Pycharms\AI\Calligraphy\DATA"
    data_path = r"/Users/william/Documents/数据集/wbblur"



    def __init__(self):
        if os.path.exists(self.x_trainI1_savepath) \
                and os.path.exists(self.x_trainI2_savepath) \
                and os.path.exists(self.x_trainI3_savepath) \
                and os.path.exists(self.x_trainI4_savepath) \
                and os.path.exists(self.y_train_savepath) \
                and os.path.exists(self.x_testI1_savepath) \
                and os.path.exists(self.x_testI2_savepath) \
                and os.path.exists(self.x_testI3_savepath) \
                and os.path.exists(self.x_testI4_savepath) \
                and os.path.exists(self.y_test_savepath) \
                and os.path.exists(self.x_validI1_savepath) \
                and os.path.exists(self.x_validI2_savepath) \
                and os.path.exists(self.x_validI3_savepath) \
                and os.path.exists(self.x_validI4_savepath) \
                and os.path.exists(self.y_valid_savepath):
            self.load_data()
        else:
            self.generate_data()
            self.load_data()

    def load_data(self):
        print('-----------Load Datasets------------')

        # 从文件读取数据集
        x_I1_train = np.load(self.x_trainI1_savepath)
        x_I2_train = np.load(self.x_trainI2_savepath)
        x_I3_train = np.load(self.x_trainI3_savepath)
        x_I4_train = np.load(self.x_trainI4_savepath)
        x_I5_train = np.load(self.x_trainI5_savepath)
        self.y_train = np.load(self.y_train_savepath)
        self.y_train = tf.convert_to_tensor(self.y_train)

        x_I1_valid = np.load(self.x_validI1_savepath)
        x_I2_valid = np.load(self.x_validI2_savepath)
        x_I3_valid = np.load(self.x_validI3_savepath)
        x_I4_valid = np.load(self.x_validI4_savepath)
        x_I5_valid = np.load(self.x_validI5_savepath)
        self.y_valid = np.load(self.y_valid_savepath)
        self.y_valid = tf.convert_to_tensor(self.y_valid)

        x_I1_test = np.load(self.x_testI1_savepath)
        x_I2_test = np.load(self.x_testI2_savepath)
        x_I3_test = np.load(self.x_testI3_savepath)
        x_I4_test = np.load(self.x_testI4_savepath)
        x_I5_test = np.load(self.x_testI5_savepath)
        self.y_test = np.load(self.y_test_savepath)
        self.y_test = tf.convert_to_tensor(self.y_test)

        self.x_I1_train = np.reshape(x_I1_train, (len(x_I1_train), 32, 32))
        self.x_I2_train = np.reshape(x_I2_train, (len(x_I2_train), 32, 32))
        self.x_I3_train = np.reshape(x_I3_train, (len(x_I3_train), 32, 32))
        self.x_I4_train = np.reshape(x_I4_train, (len(x_I4_train), 32, 32))
        self.x_I5_train = np.reshape(x_I5_train, (len(x_I5_train), 32, 32))

        self.x_I1_valid = np.reshape(x_I1_valid, (len(x_I1_valid), 32, 32))
        self.x_I2_valid = np.reshape(x_I2_valid, (len(x_I2_valid), 32, 32))
        self.x_I3_valid = np.reshape(x_I3_valid, (len(x_I3_valid), 32, 32))
        self.x_I4_valid = np.reshape(x_I4_valid, (len(x_I4_valid), 32, 32))
        self.x_I5_valid = np.reshape(x_I5_valid, (len(x_I5_valid), 32, 32))

        self.x_I1_test = np.reshape(x_I1_test, (len(x_I1_test), 32, 32))
        self.x_I2_test = np.reshape(x_I2_test, (len(x_I2_test), 32, 32))
        self.x_I3_test = np.reshape(x_I3_test, (len(x_I3_test), 32, 32))
        self.x_I4_test = np.reshape(x_I4_test, (len(x_I4_test), 32, 32))
        self.x_I5_test = np.reshape(x_I5_test, (len(x_I5_test), 32, 32))

        self.x_I1_train = tf.convert_to_tensor(x_I1_train)
        self.x_I2_train = tf.convert_to_tensor(x_I2_train)
        self.x_I3_train = tf.convert_to_tensor(x_I3_train)
        self.x_I4_train = tf.convert_to_tensor(x_I4_train)
        self.x_I5_train = tf.convert_to_tensor(x_I5_train)

        self.x_I1_test = tf.convert_to_tensor(x_I1_test)
        self.x_I2_test = tf.convert_to_tensor(x_I2_test)
        self.x_I3_test = tf.convert_to_tensor(x_I3_test)
        self.x_I4_test = tf.convert_to_tensor(x_I4_test)
        self.x_I5_test = tf.convert_to_tensor(x_I5_test)

        self.x_I1_valid = tf.convert_to_tensor(x_I1_valid)
        self.x_I2_valid = tf.convert_to_tensor(x_I2_valid)
        self.x_I3_valid = tf.convert_to_tensor(x_I3_valid)
        self.x_I4_valid = tf.convert_to_tensor(x_I4_valid)
        self.x_I5_valid = tf.convert_to_tensor(x_I5_valid)


        #增加一维，-1表示最后一维，也可以通过下标指定
        self.x_I1_train = tf.expand_dims(self.x_I1_train,-1)
        self.x_I2_train = tf.expand_dims(self.x_I2_train, -1)
        self.x_I3_train = tf.expand_dims(self.x_I3_train, -1)
        self.x_I4_train = tf.expand_dims(self.x_I4_train, -1)
        self.x_I5_train = tf.expand_dims(self.x_I5_train, -1)

        self.x_I1_valid = tf.expand_dims(self.x_I1_valid, -1)
        self.x_I2_valid = tf.expand_dims(self.x_I2_valid, -1)
        self.x_I3_valid = tf.expand_dims(self.x_I3_valid, -1)
        self.x_I4_valid = tf.expand_dims(self.x_I4_valid, -1)
        self.x_I5_valid = tf.expand_dims(self.x_I5_valid, -1)

        self.x_I1_test = tf.expand_dims(self.x_I1_test, -1)
        self.x_I2_test = tf.expand_dims(self.x_I2_test, -1)
        self.x_I3_test = tf.expand_dims(self.x_I3_test, -1)
        self.x_I4_test = tf.expand_dims(self.x_I4_test, -1)
        self.x_I5_test = tf.expand_dims(self.x_I5_test, -1)

    def generate_data(self):
        print('-----------Generate Datasets--------')
        temp_x, temp_y = self.create_data(self.data_path + '/Test', "Test")
        x_I1_test = np.array(temp_x[0])
        x_I2_test = np.array(temp_x[1])
        x_I3_test = np.array(temp_x[2])
        x_I4_test = np.array(temp_x[3])
        x_I5_test = np.array(temp_x[4])
        y_test = np.array(temp_y)

        x_I1_test, x_I2_test, x_I3_test, x_I4_test, x_I5_test, y_test = self.shuffe_data(x_I1_test, x_I2_test, x_I3_test, x_I4_test, x_I5_test, y_test)

        temp_x, temp_y = self.create_data(self.data_path + '/Train', "Train")
        x_I1_train = np.array(temp_x[0])
        x_I2_train = np.array(temp_x[1])
        x_I3_train = np.array(temp_x[2])
        x_I4_train = np.array(temp_x[3])
        x_I5_train = np.array(temp_x[4])
        y_train = np.array(temp_y)

        x_I1_train, x_I2_train, x_I3_train, x_I4_train, x_I5_train, y_train = self.shuffe_data(x_I1_train, x_I2_train,
                                                                                         x_I3_train, x_I4_train,
                                                                                         x_I5_train, y_train)

        temp_x, temp_y = self.create_data(self.data_path + '/Valid', "Valid")
        x_I1_valid = np.array(temp_x[0])
        x_I2_valid = np.array(temp_x[1])
        x_I3_valid = np.array(temp_x[2])
        x_I4_valid = np.array(temp_x[3])
        x_I5_valid = np.array(temp_x[4])
        y_valid = np.array(temp_y)

        x_I1_valid, x_I2_valid, x_I3_valid, x_I4_valid, x_I5_valid, y_valid = self.shuffe_data(x_I1_valid, x_I2_valid,
                                                                                               x_I3_valid, x_I4_valid,
                                                                                               x_I5_valid, y_valid)


        print("--------Save Datasets-------------")


        self.x_I1_train = np.reshape(x_I1_train, (len(x_I1_train), -1))
        self.x_I2_train = np.reshape(x_I2_train, (len(x_I2_train), -1))
        self.x_I3_train = np.reshape(x_I3_train, (len(x_I3_train), -1))
        self.x_I4_train = np.reshape(x_I4_train, (len(x_I4_train), -1))
        self.x_I5_train = np.reshape(x_I5_train, (len(x_I5_train), -1))

        self.x_I1_valid = np.reshape(x_I1_valid, (len(x_I1_valid), -1))
        self.x_I2_valid = np.reshape(x_I2_valid, (len(x_I2_valid), -1))
        self.x_I3_valid = np.reshape(x_I3_valid, (len(x_I3_valid), -1))
        self.x_I4_valid = np.reshape(x_I4_valid, (len(x_I4_valid), -1))
        self.x_I5_valid = np.reshape(x_I5_valid, (len(x_I5_valid), -1))

        self.x_I1_test = np.reshape(x_I1_test, (len(x_I1_test), -1))
        self.x_I2_test = np.reshape(x_I2_test, (len(x_I2_test), -1))
        self.x_I3_test = np.reshape(x_I3_test, (len(x_I3_test), -1))
        self.x_I4_test = np.reshape(x_I4_test, (len(x_I4_test), -1))
        self.x_I5_test = np.reshape(x_I5_test, (len(x_I5_test), -1))

        self.y_test = y_test
        self.y_train = y_train
        self.y_valid = y_valid

        np.save(self.x_trainI1_savepath, x_I1_train)
        np.save(self.x_trainI2_savepath, x_I2_train)
        np.save(self.x_trainI3_savepath, x_I3_train)
        np.save(self.x_trainI4_savepath, x_I4_train)
        np.save(self.x_trainI5_savepath, x_I5_train)

        np.save(self.x_validI1_savepath, x_I1_valid)
        np.save(self.x_validI2_savepath, x_I2_valid)
        np.save(self.x_validI3_savepath, x_I3_valid)
        np.save(self.x_validI4_savepath, x_I4_valid)
        np.save(self.x_validI5_savepath, x_I5_valid)

        np.save(self.x_testI1_savepath, x_I1_test)
        np.save(self.x_testI2_savepath, x_I2_test)
        np.save(self.x_testI3_savepath, x_I3_test)
        np.save(self.x_testI4_savepath, x_I4_test)
        np.save(self.x_testI5_savepath, x_I5_test)

        np.save(self.y_train_savepath, y_train)
        np.save(self.y_test_savepath, y_test)
        np.save(self.y_valid_savepath, y_valid)


    def shuffe_data(self, x_train_I1, x_train_I2, x_train_I3, x_train_I4, x_train_I5, y_train):
        index = [i for i in range(len(x_train_I1))]
        np.random.shuffle(index)
        temp_y = []
        temp_x_I1 = []
        temp_x_I2 = []
        temp_x_I3 = []
        temp_x_I4 = []
        temp_x_I5 = []
        for i in index:
            temp_y.append(y_train[i])
            temp_x_I1.append(x_train_I1[i])
            temp_x_I2.append(x_train_I2[i])
            temp_x_I3.append(x_train_I3[i])
            temp_x_I4.append(x_train_I4[i])
            temp_x_I5.append(x_train_I5[i])

        return temp_x_I1, temp_x_I2, temp_x_I3, temp_x_I4, temp_x_I5, temp_y

    def load_train(self):

        return self.x_I1_train, self.x_I2_train, self.x_I3_train, self.x_I4_train,self.x_I5_train, self.y_train

    def load_test(self):
        return self.x_I1_test, self.x_I2_test, self.x_I3_test, self.x_I4_test,self.x_I5_test, self.y_test

    def load_valid(self):
        return self.x_I1_valid, self.x_I2_valid, self.x_I3_valid, self.x_I4_valid, self.x_I5_valid,self.y_valid

    def create_data(self, path, ftlag):
        train_x1, train_x2, train_x3, train_x4, train_x5 = [], [], [], [], []
        train_y = []
        dict = {}
        try:
            # 读取数据集文件tensorflow.
            # 1. 根据数据集的划分，将数据集划分成训练集和测试集
            j = 0
            for file in os.listdir(path=path):

                tag = file  # 获取标签
                dict[j] = tag
                tag_file = path + "/" + file
                img_list = os.listdir(tag_file)
                img_list.sort()
                tag_len = len(img_list)
                # 读取数据集
                for i in range(tag_len):
                    img_path = tag_file + "/" + img_list[i]

                    img = Image.open(img_path)
                    # 如果是LCNN，则需要将图片划分四份，然后作为输入32x32x4的输入特征向量
                    # img.crop(左，上，右，下)
                    # 将图片分成四分
                    # 要注意像素点坐标从0开始，也就意味着是从0-63
                    I1 = img.crop((0.0, 0.0, 32, 32))
                    I2 = img.crop((32, 0, 64, 32))
                    I3 = img.crop((0, 32, 32, 64))
                    I4 = img.crop((32, 32, 64, 64))
                    I5 = img.resize((32, 32), Image.ANTIALIAS)

                    imgx = np.array(I1.convert('L'))
                    imgx = imgx / 255.0  # 归一化处理
                    train_x1.append(imgx)
                    imgx = np.array(I2.convert('L'))
                    imgx = imgx / 255.0  # 归一化处理
                    train_x2.append(imgx)
                    imgx = np.array(I3.convert('L'))
                    imgx = imgx / 255.0  # 归一化处理
                    train_x3.append(imgx)
                    imgx = np.array(I4.convert('L'))
                    imgx = imgx / 255.0  # 归一化处理
                    train_x4.append(imgx)
                    imgx = np.array(I5.convert('L'))
                    imgx = imgx / 255.0  # 归一化处理
                    train_x5.append(imgx)

                    train_y.append(j)
                j = j + 1
            with open(ftlag + 'mapLabel.txt', 'w') as f:
                f.write(json.dumps(dict))
        except Exception as e :
            print(e)
        return [train_x1, train_x2, train_x3, train_x4, train_x5], train_y
if __name__ == '__main__':
    test = DATA_TPCNN()