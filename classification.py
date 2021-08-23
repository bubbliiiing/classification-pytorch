import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16
from utils.utils import letterbox_image

get_model_from_name = {
    "mobilenet":mobilenet_v2,
    "resnet50":resnet50,
    "vgg16":vgg16,
}

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def _preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path和classes_path和backbone都需要修改！
#--------------------------------------------#
class Classification(object):
    _defaults = {
        "model_path"    : 'model_data/mobilenet_catvsdog.pth',
        "classes_path"  : 'model_data/cls_classes.txt',
        "input_shape"   : [224,224,3],
        "backbone"      : 'mobilenet',
        "cuda"          : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)

        # 计算总的种类
        self.num_classes = len(self.class_names)

        assert self.backbone in ["mobilenet", "resnet50", "vgg16"]

        self.model = get_model_from_name[self.backbone](num_classes=self.num_classes, pretrained=False)

        self.model = self.model.eval()
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        print('{} model, and classes loaded.'.format(model_path))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        old_image = copy.deepcopy(image)
        
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        photo = np.array(crop_img,dtype = np.float32)

        # 图片预处理，归一化
        photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        photo = np.transpose(photo,(0,3,1,2))

        with torch.no_grad():
            photo = Variable(torch.from_numpy(photo).type(torch.FloatTensor))
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        plt.subplot(1, 1, 1)
        plt.imshow(np.array(old_image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        plt.show()
        return class_name

    def close_session(self):
        self.sess.close()
