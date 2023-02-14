import numpy as np
from PIL import Image
import itertools
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_matrix(cm, class_num, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    matrix图
    -->cm = np.array(cm)
    -->classes = ['Liu Gongquan', 'Ouyang Xun', 'Yan Zhen', 'Zhao Mengfu']
    -->plot_matrix(cm, classes, normalize=True)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix')
    else:
        print('Confusin matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title:
        plt.title(title, fontsize=20)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(class_num))
    plt.axis('equal')
    ax = plt.gca()
    l, r = plt.xlim()
    ax.spines['left'].set_position(('data', l))
    ax.spines['right'].set_position(('data', r))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor('white')
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = float('{:.4f}'.format(cm[i, j])) if normalize else int(cm[i, j])
        # print(type(num))
        plt.text(
            j, i, num,
            verticalalignment='center',
            horizontalalignment='center',
            color='white' if num > thresh else 'black',
            fontsize=18
        )
    plt.tight_layout()
    plt.xticks(tick_marks, class_num, rotation=0, fontsize=10)
    plt.yticks(tick_marks, class_num, rotation=0, fontsize=10)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.show()
    # plt.savefig(savname)


def plot_k(k, acc, log_dir):
    """
    画权重-acc折线图
    """

    plt.figure()
    plt.plot(k, acc, 'red', linewidth=2, label='acc')

    plt.grid(True)
    plt.xlabel('λ', fontsize=14)
    plt.ylabel('Accuracy(%)', fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(log_dir, "k_acc.png"))
    plt.cla()
    plt.close("all")


def plot_cm(classes, matrix, savname):
    """
    classes: a list of class names
    画混淆矩阵
    -->conf_matrix = test(model, loss_func)
    -->classes = ('ARB', 'BEN', 'ENG', 'GUJ', 'HIN', 'KAN', 'ORI', 'PUN', 'TAM', 'TEL')
    -->plotCM(list(classes), conf_matrix, "Confusion_Matrix_cvsi.jpeg")
    """
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    # plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), fontdict={'fontsize': 8}, va='center', ha='center')
    ax.set_xticklabels([''] + classes, fontdict={'fontsize': 10}, rotation=90)
    ax.set_yticklabels([''] + classes)
    l, r = plt.xlim()
    ax.spines['left'].set_position(('data', l))
    ax.spines['right'].set_position(('data', r))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor('white')
    # save
    # plt.savefig(savname)
    plt.show()


def get_max_acc(file):
    """
    获取前五并求平均
    :param file: acc文件
    :return: max
    """
    with open(file=file) as f:
        lines = f.readlines()
    ordered_lines = sorted(lines, reverse=True)
    nums = []
    for i in ordered_lines:
        s = i.strip()
        nums.append(float(s))
    top_5 = nums[0] + nums[1] + nums[2] + nums[3] + nums[4]
    max_num = top_5 / 5
    return max_num


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

    # ---------------------------------------------------#


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


def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


if __name__ == '__main__':
    # baseline = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/baseline/epoch_val_acc.txt')
    baseline = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/16/epoch_val_acc.txt')
    # baseline_bap = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/baseline_bap/epoch_val_acc.txt')
    baseline_bap = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/32/epoch_val_acc.txt')
    # baseline_cbam = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/baseline_cbam/epoch_val_acc.txt')
    baseline_cbam = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/64/epoch_val_acc.txt')
    # baseline_cbam_bap = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/baseline_cbam_bap/epoch_val_acc.txt')
    baseline_cbam_bap = get_max_acc(r'/Users/william/Documents/GitHub/facenet/logs/128/epoch_val_acc.txt')
    print(baseline, baseline_bap, baseline_cbam, baseline_cbam_bap)
