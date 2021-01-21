from __future__ import print_function
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, fdir, model_name):
    filepath = os.path.join(fdir, model_name + '.tar')
    if is_best:
        torch.save(state, filepath)
        print('[info] save best model')


def save_history(cls_orig_acc, clease_trig_acc, cls_trig_loss, at_trig_loss, at_epoch_list, logs_dir):
    dataframe = pd.DataFrame({'epoch': at_epoch_list, 'cls_orig_acc': cls_orig_acc, 'clease_trig_acc': clease_trig_acc,
                              'cls_trig_loss': cls_trig_loss, 'at_trig_loss': at_trig_loss})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(logs_dir, index=False, sep=',')

def plot_curve(clean_acc, bad_acc, epochs, dataset_name):
    N = epochs+1
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), clean_acc, label="Classification Accuracy", marker='D', color='blue')
    plt.plot(np.arange(0, N), bad_acc, label="Attack Success Rate",  marker='o', color='red')
    plt.title(dataset_name)
    plt.xlabel("Epoch")
    plt.ylabel("Student Model Accuracy/Attack Success Rate(%)")
    plt.xticks(range(0, N, 1))
    plt.yticks(range(0, 101, 20))
    plt.legend()
    plt.show()