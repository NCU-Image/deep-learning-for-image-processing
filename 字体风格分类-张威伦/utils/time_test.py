import numpy as np
import torch




def infer_time():
    """
    测试单张图片的预测时间
    :return:
    """
    num_classes = 4
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
    model = Facenet(backbone=backbone, mode="predict", num_classes=num_classes)
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "C:\\project\\pth\\1800_4_best\\ep098-loss0.008-val_loss0.171.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.eval()
    model = model.cuda()
    dummy_input = torch.randn(1, 3, 160, 160, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP：开始跑dummy example
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
