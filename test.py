from torchsummary import summary

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16

if __name__ == "__main__":
    model = vgg16(num_classes=1000, pretrained=False).train().cuda()
    summary(model,(3,473,473))
