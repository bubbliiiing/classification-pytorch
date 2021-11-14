from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16

get_model_from_name = {
    "mobilenet"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
}