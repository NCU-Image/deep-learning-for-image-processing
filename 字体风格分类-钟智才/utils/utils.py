import argparse
def step_decay(epoch):
    """
    动态调整学习率
    :param epoch: 训练轮数
    :return: 返回学习率
    """
    if epoch < 30 :
        lrate =  1e-3
    elif epoch < 90:
        lrate = 1e-4
    else:
        lrate = 1e-5
    return lrate

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument("--test",type=)
