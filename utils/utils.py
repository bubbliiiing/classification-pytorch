import math
from functools import partial

import numpy as np
import torch
from PIL import Image

from .utils_aug import resize, center_crop


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    # x /= 127.5
    # x -= 1.
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vit_b_16': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/vit-patch_16.pth',
        'swin_transformer_tiny': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_tiny_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_small': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_small_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_base': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_base_patch4_window7_224_imagenet1k.pth'
    }
    try:
        url = download_urls[backbone]
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        load_state_dict_from_url(url, model_dir)
    except:
        print("There is no pretrained model for " + backbone)