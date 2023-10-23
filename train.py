import torch
import torch.nn as nn
import torch.nn.functional as F
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing #0.9
        self.smoothing = smoothing #0.1
        self.cls = classes  #54
        self.dim = dim
        self.true_dist = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim) #对输出进行softmax
        # print('pred.shape', pred.shape)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)  #0列表
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1).cuda(), self.confidence)
            # true_dist.cpu().scatter_(1, target.unsqueeze(1).cpu(), self.confidence)

        self.true_dist = true_dist
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
#举例：update(loss,batch_size),  val为loss为一个batch损失的均值， sum统计一个batch整体损失， count记录样本数量， avg记录总的损失均值
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, label):
    _,pred = output.topk(1,1)
    batch_size = len(pred)
    pred = pred.view(-1,batch_size).squeeze()
    compare = pred.eq(label)

    correct = compare.sum()*1.0
    res = []
    res.append(correct*100.0/batch_size)

    return res


def train(data_loader, model, criterion, optimizer, epoch, accumulation_steps):
    losses = AverageMeter()
    # top = AverageMeter()

    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    num = 0

    # 进来一个batch的数据，计算一次梯度，更新一次网络
    for i, (img, target) in enumerate(data_loader):
        img = img.cuda()
        target = target.cuda()
        batch_size = target.shape[0]

        # vec, s, output = model(img)
        vec, output = model(img)
        loss = criterion(output, target)
        # print(output)
        # print(target)

        acc = accuracy(output, target)
        epoch_loss += loss
        losses.update(loss.item(), batch_size)
        loss.backward()

        num += 1
        if((i+1)%accumulation_steps == 0):
            optimizer.step()
            optimizer.zero_grad()

        if(i%10==0):
            print('Alzhehimer - 3DCNN <==> Train Epoch: [{0}][{1}/{2}] Acc: {3}\n'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, i, len(data_loader), acc[0], loss=losses))


    if(num % accumulation_steps):
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss