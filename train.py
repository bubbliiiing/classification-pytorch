import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import get_model_from_name

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes, weights_init
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #----------------------------------------------------#
    Cuda            = True
    #----------------------------------------------------#
    #   训练自己的数据集的时候一定要注意修改classes_path
    #   修改成自己对应的种类的txt
    #----------------------------------------------------#
    classes_path    = 'model_data/cls_classes.txt' 
    #----------------------------------------------------#
    #   输入的图片大小
    #----------------------------------------------------#
    input_shape     = [224, 224]
    #----------------------------------------------------#
    #   所用模型种类：
    #   mobilenet、resnet50、vgg16、vit
    #
    #   在使用vit时学习率需要设置的小一些，否则不收敛
    #   可以将最下方的两个lr分别设置成1e-4、1e-5
    #----------------------------------------------------#
    backbone        = "mobilenet"
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ""
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   获得图片路径和标签
    #------------------------------------------------------#
    annotation_path = "cls_train.txt"
    #------------------------------------------------------#
    #   进行训练集和验证集的划分，默认使用10%的数据用于验证
    #------------------------------------------------------#
    val_split       = 0.1
    #------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------#
    num_workers     = 4

    #------------------------------------------------------#
    #   获取classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if backbone != "vit":
        model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
        
    loss_history = LossHistory("logs/")
    #----------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val

    #------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为1。
    #   
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch为总训练世代
    #   提示OOM或者显存不足请调小batch_size
    #------------------------------------------------------#
    if True:
        #----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        #----------------------------------------------------#
        lr              = 1e-3
        Batch_size      = 32
        Init_Epoch      = 0
        Freeze_Epoch    = 50

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = DataGenerator(lines[:num_train], input_shape, True)
        val_dataset     = DataGenerator(lines[num_train:], input_shape, False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        #----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        #----------------------------------------------------#
        lr              = 1e-4
        Batch_size      = 16
        Freeze_Epoch    = 50
        Epoch           = 100

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = DataGenerator(lines[:num_train], input_shape, True)
        val_dataset     = DataGenerator(lines[num_train:], input_shape, False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        if Freeze_Train:
            model.Unfreeze_backbone()

        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, Cuda)
            lr_scheduler.step()
