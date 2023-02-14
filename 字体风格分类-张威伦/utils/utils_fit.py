import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.paper_utils import get_lr


def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, epoch_step_test
                  , gen, gen_val, gen_test, Epoch, cuda, batch_size, save_period, save_dir):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_CE_loss = 0
    val_total_accuracy = 0

    test_total_CE_loss = 0
    test_total_accuracy = 0

    print('Start Train \n')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

        optimizer.zero_grad()
        outputs1, outputs2 = model_train(images, "train")
        _triplet_loss = loss(outputs1, batch_size)
        _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
        _loss = 0.6 * _triplet_loss + _CE_loss
        _loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss += _triplet_loss.item()
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                            'total_CE_loss': total_CE_loss / (iteration + 1),
                            'accuracy': total_accuracy / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

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
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs1 = model_train(images, mode='predict')

            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs1, dim=-1), labels)
            _loss = _CE_loss
            argmax = torch.argmax(F.softmax(outputs1, dim=-1), dim=-1)
            total_acc = (argmax == labels).type(torch.FloatTensor)
            accuracy = torch.mean(total_acc)

            val_total_CE_loss += _CE_loss.item()
            val_total_accuracy += accuracy.item()

        pbar.set_postfix(**{'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                            'val_accuracy': val_total_accuracy / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    pbar.close()
    print('Finish Validation \n')

    model_train.eval()
    print('Start Test \n')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for iteration, (images, labels) in enumerate(gen_test):
        if iteration >= epoch_step_test:
            break
        with torch.no_grad():
            images = images.type(torch.FloatTensor)
            labels = labels.type(torch.int64)
            if cuda:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs1 = model_train(images, mode='predict')

            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs1, dim=-1), labels)
            _loss = _CE_loss
            argmax = torch.argmax(F.softmax(outputs1, dim=-1), dim=-1)
            total_acc = (argmax == labels).type(torch.FloatTensor)
            accuracy = torch.mean(total_acc)

            test_total_CE_loss += _CE_loss.item()
            test_total_accuracy += accuracy.item()

        pbar.set_postfix(**{'test_total_CE_loss': test_total_CE_loss / (iteration + 1),
                            'test_accuracy': test_total_accuracy / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)

    loss_history.append_loss(epoch, test_total_accuracy / epoch_step_test,
                             val_total_accuracy / epoch_step_val,
                             (total_triple_loss + total_CE_loss) / epoch_step,
                             val_total_CE_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(),
                   os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % ((epoch + 1),
                                                                                (
                                                                                        total_triple_loss + total_CE_loss) / epoch_step,
                                                                                val_total_CE_loss / epoch_step_val)))
