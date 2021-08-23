from random import shuffle

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.utils import letterbox_image


#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def _preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
    image = image.convert("RGB")
    h, w = input_shape

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.75, 1.25)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    rotate = rand()<.5
    if rotate: 
        angle=np.random.randint(-15,15)
        a,b=w/2,h/2
        M=cv2.getRotationMatrix2D((a,b),angle,1)
        image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[128,128,128]) 

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:,:, 0]>360, 0] = 360
    x[:, :, 1:][x[:, :, 1:]>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
    return image_data

class DataGenerator(data.Dataset):
    def __init__(self, input_shape, lines, random=True):
        self.input_shape = input_shape
        self.lines = lines
        self.random = random

    def __len__(self):
        return len(self.lines)

    def get_len(self):
        return len(self.lines)

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.lines)

        annotation_path = self.lines[index].split(';')[1].split()[0]
        img = Image.open(annotation_path)
        
        if self.random:
            img = get_random_data(img, [self.input_shape[0],self.input_shape[1]])
        else:
            img = letterbox_image(img, [self.input_shape[0],self.input_shape[1]])

        img = np.array(img).astype(np.float32)
        img = _preprocess_input(img)
        img = np.transpose(img,[2,0,1])

        y = int(self.lines[index].split(';')[0])
        return img, y

def detection_collate(batch):
    images = []
    targets = []
    for img, y in batch:
        images.append(img)
        targets.append(y)
    images = np.array(images)
    targets = np.array(targets)
    return images, targets
