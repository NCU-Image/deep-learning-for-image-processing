import csv
import datetime
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
# 读取csv文件，返回训练次数
def get_init_epoch(filename):
    with open(filename) as f:
        f_csv = csv.DictReader(f)
        count = 0
        for row in f_csv:
            count = count+1
        return count

# 读取csv文件，返回训练数据

def get_data_list(filename):
    epoch,train_acc,train_loss,val_acc,val_loss = [],[],[],[],[]
    # 利用pandas读取csv文件
    data = pd.read_csv(filename,engine='python') # 防止因为中文报错
    epoch = data['epoch'].values.tolist() # 读取训练轮数
    train_acc = data['sparse_categorical_accuracy'].values.tolist() #读取训练准确率
    train_loss = data['loss'].values.tolist() #读取训练损失
    val_acc = data['val_sparse_categorical_accuracy'].values.tolist()# 读取验证准确率
    val_loss = data['val_loss'].values.tolist() # 读取验证loss
    return epoch,train_acc,train_loss,val_acc,val_loss


# 可视化训练数据,通过plt
def show_data_plt(filename):
    epoch, train_acc, train_loss, val_acc, val_loss = get_data_list(filename)
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# 将数据转化为Tensorboard的日志文件
def show_tensorboard(filename,dirname):
    # 获取结果数据集
    epoch, train_acc, train_loss, val_acc, val_loss = get_data_list(filename)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间戳
    train_log_dir = dirname+'logs/gradient_tape/' + current_time + '/train' # 训练数据
    test_log_dir = dirname+'logs/gradient_tape/' + current_time + '/test' # 测试数据
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for i in range(len(epoch)):
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss[i], step=i)
            tf.summary.scalar('accuracy', train_acc[i], step=i)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss[i], step=i)
            tf.summary.scalar('accuracy', val_acc[i], step=i)

if __name__ == '__main__':
    #show_data_plt(r"E:\Pycharms\AI\Calligraphy\firstDATA\LCNN\trainvalid_test\1\training_log")

    show_data_plt(r"E:\Pycharms\AI\Calligraphy\firstDATA\GCNN\trainvalid_test\1\training_log")
    # show_data_plt(r"E:\Pycharms\AI\Calligraphy\firstDATA\TPCNN\train_valid\2\training_log")
    # show_data_plt(r"E:\Pycharms\AI\Calligraphy\firstDATA\TPCNN\train_valid\3\training_log")