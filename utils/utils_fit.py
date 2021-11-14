import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            images, targets = batch
            with torch.no_grad():
                images      = torch.from_numpy(images).type(torch.FloatTensor)
                targets     = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()
            outputs     = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs     = model_train(images)
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
                
                val_loss    += loss_value.item()
                
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
