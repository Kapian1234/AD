import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
#举例：update(loss,batch_size),  val为loss为一个batch损失的均值， sum统计一个batch整体损失， count记录一个样本数量， avg记录总的损失
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top = AverageMeter()

    #不启动BN和Dropout
    model.eval()

    true_num = 0
    total_num = 0
    sen_num = 0
    spe_num = 0
    true_sen_num = 0
    true_spe_num = 0
    # we may have ten d in data
    with torch.no_grad():
        for num, (img, target) in enumerate(val_loader):

            img = img.cuda()
            target = target.cuda()
            batch_size = target.shape[0]

            _, output = model(img)
            loss = criterion(output, target)


            list_target = target.tolist()
            for i in list_target:
                if i == 0:
                    sen_num += 1
                else:
                    spe_num += 1
            _,pred = torch.max(output, 1)

            losses.update(loss.item(), batch_size)

            for i in range(len(pred)):
                if pred[i] == target[i]:
                    true_num += 1
                    if target[i] == 0:
                        true_sen_num += 1
                    else:
                        true_spe_num += 1
            total_num += batch_size


    print('正确数目： ',true_num,'  总数目： ',total_num)
    print('sen_num: ',sen_num,'  true_sen_num: ', true_sen_num, '  spe_num: ', spe_num,'  true_spe_num: ',true_spe_num)
    acc = true_num/total_num
    sen = true_sen_num/sen_num  # recall
    spe = true_spe_num/spe_num

    precison = true_sen_num / (true_sen_num + (sen_num - true_sen_num))
    recall = sen

    f1_score = 2*precison*recall / (precison+recall)

    print('验证准确率\n',acc, '\n敏感性\n', sen, '\n特异性\n', spe, '\nF1-Score\n', f1_score)

    return [acc, sen, spe, f1_score]
