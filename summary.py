#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from torchsummary import summary

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16
from nets.vit import vit

if __name__ == "__main__":
    model = mobilenet_v2(num_classes=1000, pretrained=False).train().cuda()
    summary(model,(3, 224, 224))
