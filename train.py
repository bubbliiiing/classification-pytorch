import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.mobilenet import mobilenet_v2
from nets.resnet50 import resnet50
from nets.vgg16 import vgg16
from utils.utils import weights_init
from utils.dataloader import DataGenerator, detection_collate

get_model_from_name = {
    "mobilenet" : mobilenet_v2,
    "resnet50"  : resnet50,
    "vgg16"     : vgg16,
}

freeze_layers = {
    "mobilenet" :81,
    "resnet50"  :173,
    "vgg16"     :19,
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    total_accuracy = 0

    val_toal_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size: 
                break
            images, targets = batch
            with torch.no_grad():
                images      = torch.from_numpy(images).type(torch.FloatTensor)
                targets     = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()

            outputs = net(images)
            loss    = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()
    
    
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs = net(images)
                val_loss = nn.CrossEntropyLoss()(outputs, targets)
                
                val_toal_loss += val_loss.item()
                
            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#----------------------------------------#
#   主函数
#----------------------------------------#
if __name__ == "__main__":
    log_dir = "./logs/"
    #---------------------#
    #   所用模型种类
    #---------------------#
    backbone = "mobilenet"
    #---------------------#
    #   输入的图片大小
    #---------------------#
    input_shape = [224,224,3]
    #-------------------------------#
    #   Cuda的使用
    #-------------------------------#
    Cuda = True

    #-------------------------------#
    #   是否使用网络的imagenet
    #   预训练权重
    #-------------------------------#
    pretrained = True

    classes_path = './model_data/cls_classes.txt' 
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    assert backbone in ["mobilenet", "resnet50", "vgg16"]

    model = get_model_from_name[backbone](num_classes=num_classes,pretrained=pretrained)
    if not pretrained:
        weights_init(model)

    #------------------------------------------#
    #   注释部分可用于断点续练
    #   将训练好的模型重新载入
    #------------------------------------------#
    # # 加快模型训练的效率
    # model_path = "model_data/Omniglot_vgg.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    with open(r"./cls_train.txt","r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-3
        Batch_size      = 32
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        
        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        model.freeze_backbone()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr              = 1e-4
        Batch_size      = 16
        Freeze_Epoch    = 50
        Epoch           = 100

        optimizer       = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_dataset   = DataGenerator(input_shape,lines[:num_train])
        val_dataset     = DataGenerator(input_shape,lines[num_train:], False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)

        epoch_size      = train_dataset.get_len()//Batch_size
        epoch_size_val  = val_dataset.get_len()//Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda)
            lr_scheduler.step()
