import torch
# from model import GoogLeNet
from googlenet_enhance import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_transform = transforms.Compose(
    [ transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image

file_pathname = ("E:/py-project/deep-learning-for-image-processing-master/pytorch_classification/test_2008_head")
    #遍历该目录下的所有图片文件
all=0
dui =0
cuo =0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# create model
model = GoogLeNet(num_classes=2, aux_logits=False)
# load model weights
model_weight_path = "./googleNet_enhance.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
model.to(device)
for filename in os.listdir(file_pathname):
    img = plt.imread(file_pathname+'/'+filename)
    all+=1
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device)))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).detach().cpu().numpy()
    print(class_indict[str(predict_cla)])
    print(filename)
    lei = filename[8]

    # print(lei)
    if lei == class_indict[str(predict_cla)]:
        dui += 1
    else:
        cuo += 1

print("正确个数:", dui, "错误个数：", cuo)
accuracy = int(dui) / int(all)
print("测试总数:", all, "准确率：", accuracy)

# plt.show()
