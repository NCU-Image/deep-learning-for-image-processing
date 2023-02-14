import itertools
import os

import numpy as np
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
