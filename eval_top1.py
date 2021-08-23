import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from classification import Classification, _preprocess_input
from utils.utils import letterbox_image


class top1_Classification(Classification):
    def detect_image(self, image):        
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

        arg_pred = np.argmax(preds)
        return arg_pred

def evaluteTop1(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += pred == y
        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    return correct / total

classfication = top1_Classification()
with open(r"./cls_test.txt","r") as f:
    lines = f.readlines()
top1 = evaluteTop1(classfication, lines)
print("top-1 accuracy = %.2f%%" % (top1*100))
