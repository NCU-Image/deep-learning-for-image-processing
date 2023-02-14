import numpy as np
import torch

import utils
import os
import utils


def plot_k():
    y = [99.27, 99.44, 99.58, 99.41, 99.50, 99.72, 98.86, 99.16, 99.05, 99.02]
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    utils.plot_k(x, y, "/Users/william/Documents/GitHub/facenet/result_img")


def print_acc():
    accs = []
    folder = "/Users/william/PycharmProjects/fly_net/logs/1800_baseline/epoch_acc.txt"
    dir = os.listdir(folder)
    # dir.remove(".DS_Store")
    for item in dir:
        path = os.path.join(item, "epoch_acc.txt")
        path = os.path.join(folder, path)
        acc = utils.get_max_acc(path)
        accs.append(acc)
        print(path, acc)
    print(accs)


if __name__ == '__main__':
    print_acc()
