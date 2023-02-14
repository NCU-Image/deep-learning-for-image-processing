import torch
import torch.backends.cudnn as cudnn
import utils
from utils.dataloader import TestDataSet
from utils.utils_metrics import test
import nets
if __name__ == "__main__":
    # --------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # --------------------------------------#
    cuda = True
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #   inception_resnetv1
    #   paper
    # --------------------------------------#
    backbone = "paper"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    input_shape = [160, 160, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    model_path = "C:\\project\\pth\\1800_4_best\\ep098-loss0.008-val_loss0.171.pth"
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    # --------------------------------------#
    annotation_path_test = "cls_test.txt"
    num_classes = utils.get_num_classes(annotation_path_test)
# --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = 128

    log_interval = 1

    with open(annotation_path_test, "r") as f:
        lines = f.readlines()
    epoch_step_test = len(lines) // batch_size

    test_loader = torch.utils.data.DataLoader(
        TestDataSet(input_shape, lines, num_classes, random=True), batch_size=batch_size, shuffle=False)

    model = nets.Facenet(backbone=backbone, mode="predict", num_classes=num_classes)

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()
    model = model.cuda()

    test(test_loader, model, log_interval, batch_size, cuda, num_classes, epoch_step_test)
