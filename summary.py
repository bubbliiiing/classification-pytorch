#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets import get_model_from_name

if __name__ == "__main__":
    input_shape = [224, 224]
    num_classes = 1000
    backbone    = "mobilenetv2"
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = get_model_from_name[backbone](num_classes=num_classes, pretrained=False).to(device)
    
    summary(model, (3, input_shape[0], input_shape[1]))

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
