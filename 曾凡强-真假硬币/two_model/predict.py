import torch
from googlenet_enhance import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import os
import cv2
from vggmodel import vgg
os.environ['KMP_DUPLICATE_LIB_OK']='True'
data_transform = transforms.Compose(
    [ transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image

file_pathname = ("E:/py-project/deep-learning-for-image-processing-master/pytorch_classification/test_2008_head")
    #遍历该目录下的所有图片文件
file_pathname2 = ("E:/py-project/deep-learning-for-image-processing-master/pytorch_classification/test_2008_word")
    #遍历该目录下的所有图片文件
all=0
dui =0
cuo =0
modelone=[]
modeltwo=[]
modelmix=[]
data=[]
#google模型
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
        json_file = open('./class_indices1.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    # create model
    model = GoogLeNet(num_classes=2, aux_logits=False)
    # load model weights
    model_weight_path = "./googleNet_enhance.pth"
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)])
    print(filename)
    lei = filename[8]
    data.append(lei)

    modelone.append(class_indict[str(predict_cla)])

    # print(lei)
    # if lei == class_indict[str(predict_cla)]:
    #     dui += 1
    # else:
    #     cuo += 1
    #vgg模型
for filename in os.listdir(file_pathname2):
    img = plt.imread(file_pathname2+'/'+filename)
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

    # create model
    model = vgg(model_name="vgg16", num_classes=2)
    # load model weights
    model_weight_path = "./vgg16Net.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)], predict[predict_cla].item())
    print(filename)
    lei = filename[8]
    print(lei)
    # if lei==class_indict[str(predict_cla)]:
    #     dui+=1
    # else:
    #     cuo+=1
    modeltwo.append(class_indict[str(predict_cla)])

len = len(modelone)
for i in range(len):
    if modelone[i] == '1' and  modeltwo[i] == '1' :
        modelmix.append('1')
    elif modelone[i] == '2'  and  modeltwo[i]  =='2':
        modelmix.append('2')
    elif modelone[i] == '1 ' and  modeltwo[i]  =='2':
        modelmix.append('1')
    elif modelone[i] == '2'  and  modeltwo[i]  == '1':
        modelmix.append('1')

print(modelmix)
print(modelone)
print(modeltwo)
print(data)

#保存为txt文件
#第一个模型
file= open('modelone.txt', 'w')
for fp in modelone:
    file.write(str(fp))
    file.write('\n')
file.close()

#第二个模型
file= open('modeltwo.txt', 'w')
for fp in modeltwo:
    file.write(str(fp))
    file.write('\n')
file.close()
#判断结果
file= open('modelmix.txt', 'w')
for fp in modelmix:
    file.write(str(fp))
    file.write('\n')
file.close()
#真实数据列表
file= open('data.txt', 'w')
for fp in data:
    file.write(str(fp))
    file.write('\n')
file.close()

#
for i in range(len):
    if modelmix[i] == data[i]:
        dui +=1
    else:
        cuo +=1
#


print("正确个数:", dui, "错误个数：", cuo)
accuracy = int(dui) / int(len)
print("测试总数:", len, "准确率：", accuracy)

# plt.show()
# file=open('log.txt', 'r')
# list_read = file.readlines()
# print(list_read[6])