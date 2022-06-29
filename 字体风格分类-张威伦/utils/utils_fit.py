import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import evaluate


def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, Batch_size, save_period, save_dir, local_rank):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_CE_loss = 0
    val_total_accuracy = 0

    if local_rank == 0:
        print('Start Train \n')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()
        outputs1, outputs2 = model_train(images, "train")
        _triplet_loss = loss(outputs1, Batch_size)
        _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
        _loss = _triplet_loss + _CE_loss
        _loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train \n')
        print('Start Validation \n')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, (images, labels) in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        with torch.no_grad():
            images = images.type(torch.FloatTensor)
            labels = labels.type(torch.int64)
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs1 = model_train(images)

            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs1, dim=-1), labels)
            _loss = _CE_loss

            accuracy = torch.mean((torch.argmax(F.softmax(outputs1, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            val_total_CE_loss += _CE_loss.item()
            val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # if lfw_eval_flag:
    #     print("开始进行测试集的验证。\n")
    #     size = 0
    #     for iteration, (images, labels) in enumerate(test_loader):
    #         with torch.no_grad():
    #             images = images.type(torch.FloatTensor)
    #             labels = labels.type(torch.int64)
    #             if cuda:
    #                 images = images.cuda(local_rank)
    #                 labels = labels.cuda(local_rank)
    #             images = model_train(images)

    #             _CE_loss = nn.NLLLoss()(F.log_softmax(images, dim=-1), labels)
    #             _loss = _CE_loss

    #             accuracy = torch.mean((torch.argmax(F.softmax(images, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

    #             test_total_CE_loss += _CE_loss.item()
    #             test_total_accuracy += accuracy.item()

    #             size = iteration + 1

    #     print('Accuracy: %2.5f' % (test_total_accuracy / size))


    if local_rank == 0:
        pbar.close()
        print('Finish Validation \n')
        # loss_history.append_loss(epoch, val_total_accuracy / epoch_step_val,
        #                          (total_triple_loss + total_CE_loss) / epoch_step,
        #                          (val_total_CE_loss) / epoch_step_val)
        loss_history.append_loss(epoch, val_total_accuracy / epoch_step_val,
                                 (total_triple_loss + total_CE_loss) / epoch_step,
                                 (val_total_CE_loss) / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), 
            os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
            (total_triple_loss + total_CE_loss) / epoch_step,(val_total_CE_loss) / epoch_step_val)))
