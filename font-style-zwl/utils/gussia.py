import os

from torchvision import transforms
import torch
from PIL import Image, ImageFilter
import uuid



def gaussian(img, mean, std):
    c, h, w = img.shape
    noise = torch.randn([c, h, w]) * std + mean
    return noise


def gaussian_noise(input_img: str, out_img: str, std: float):
    img_jpg = Image.open(input_img).convert('RGB')

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_jpg)

    noise_tensor = gaussian(img_tensor, 0, std)
    noise_img_tensor = img_tensor + noise_tensor
    for i in range(img_tensor.shape[0]):  # min-max normalization
        noise_tensor[i] = (noise_tensor[i] - noise_tensor[i].min()) / (noise_tensor[i].max() - noise_tensor[i].min())
        noise_img_tensor[i] = (noise_img_tensor[i] - noise_img_tensor[i].min()) / (
                noise_img_tensor[i].max() - noise_img_tensor[i].min())

    to_PILimage = transforms.ToPILImage()
    noise = to_PILimage(noise_tensor)
    noise_img = to_PILimage(noise_img_tensor)

    # noise.save('/Users/william/Documents/GitHub/facenet/utils/2.bmp')
    noise_img.save(out_img)

    print('Done.')


def gaussian_blur(input_img: str, out_img: str, radio: float):
    """
    高斯模糊
    :param radio:
    :param input_img:
    :param out_img:
    :return:
    """
    img = Image.open(input_img)
    img3 = img.filter(ImageFilter.GaussianBlur(radio))
    img3.show()
    img3.save(out_img, quality=100)


def gen_new_dataset(input_dir: str):
    out_dir = '/Users/william/Documents/数据集/wb_william'
    g = os.walk(input_dir)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            img = os.path.join(path, file_name)
            if "DS_Store" in img:
                continue
            print(img)
            author = img.split("/")[6]
            uuid__hex = uuid.uuid1().hex
            gaussian_noise(img, out_dir + f'/{author}/{uuid__hex}', 0.005)
            gaussian_noise(img, out_dir + f'/{author}/{uuid__hex}', 0.05)
            gaussian_noise(img, out_dir + f'/{author}/{uuid__hex}', 0.5)
            gaussian_blur(img, out_dir + f'/{author}/{uuid__hex}', 1.5)
            gaussian_blur(img, out_dir + f'/{author}/{uuid__hex}', 1)
            gaussian_blur(img, out_dir + f'/{author}/{uuid__hex}', 0.5)


if __name__ == '__main__':
    input_dir = '/Users/william/Documents/数据集/wb'
    gen_new_dataset(input_dir)
    # input_img = '/Users/william/Documents/GitHub/facenet/utils/1.bmp'
    # out_img = '/Users/william/Documents/GitHub/facenet/utils/3.bmp'
    # gaussian_blur(input_img, out_img)
