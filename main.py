import os
import ADmodel
import loader
import torch
import val
import train


class Params:
    epochs = 2
    start_epoch = 0

args = Params()


class Trainer():
    def __init__(self, model):
        self.model = model
        self.phases = ['train', 'val', 'test']
        self.csv_path = {'train': train_df_path + 'train.csv',
                         'val': train_df_path + 'test.csv',
                         'test': train_df_path + 'test.csv'}
        # batch大小
        self.batch_size = {'train': 4, 'val': 3, 'test': 3}
        # 训练轮数
        self.epochs = args.epochs

        # 损失函数
        self.criterion = train.LabelSmoothingLoss(2, 0.05)
        # 初始学习速率
        self.lr = 0.001

        # 累积步数
        self.accumulation_steps = 8

        # 优化器
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.05)  # 5e-3

        # lr调节器
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.7, mode="min", patience=2, verbose=True)

        # 不同阶段的dataloader
        self.dataloader = {
            phase : loader.newADdataloader(phase, csv_path=self.csv_path[phase], batch_size=self.batch_size[phase])
            for phase in self.phases
        }

    def test(self):
        prec = val.validate(self.dataloader['test'], self.model, self.criterion)
        return prec

    def start(self, i):
        losses = []
        best_val_acc = 0

        for epoch in range(self.epochs):
            # 训练
            loss = train.train(self.dataloader['train'], self.model,self.criterion, self.optimizer, epoch, self.accumulation_steps)
            # 验证
            val_result = val.validate(self.dataloader['val'], self.model, self.criterion)
            val_acc = round(val_result[0],4)
            val_sen = round(val_result[1],4)
            val_spe = round(val_result[2],4)
            # 调整lr
            self.lr_scheduler.step(val_acc)
            best_val_acc = max(val_acc, best_val_acc)
            # 保存
            state = {'epoch':epoch, 'loss':loss, 'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}
            s_path = save_path + 'f' + str(i) + '/' + str(epoch)+'_'+ str(val_acc)+'_'+ str(val_sen)+'_'+ str(val_spe)+'.pth'
            if not os.path.exists(save_path + 'f' + str(i) + '/'):
                os.makedirs(save_path + 'f' + str(i) + '/')
            print(s_path)
            # torch.save(state,'./para/'+ type + '/' + 'fold' + str(i) +'-'+str(prec1)+'.pth')
            torch.save(state, s_path)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csv_path = './folds'
    data_path = './cn_pre/pre'
    save_path = './save/'

    # 训练并保存模型
    for i in range(5):
        train_df_path = csv_path + '/' + 'fold' + str(i + 1) + '/'
        model = ADmodel.Model().cuda()
        model_train = Trainer(model)
        model_train.start(i + 1)
        del model


    for j in range(5):
        l_prec = [] # 每个模型精度
        l_best = []
        path = './save' + '/f'+str(j+1)+'/'
        train_df_path = csv_path + '/' + 'fold' + str(j + 1) + '/'
        path_list = os.listdir(path)

        for i in range(len(path_list)):
            # 检查点?
            ckpt_path = path + path_list[i]
            l_best.append(ckpt_path)
            print(ckpt_path)
            # 加载保存好的模型
            model = ADmodel.Model().cuda()
            state = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(0))
            model.load_state_dict(state['state_dict'], strict=True)
            model_train = Trainer(model)
            # 测试
            print('test start\n')
            prec = model_train.test()
            l_prec.append(prec)
            del model
        print(l_prec)
        print(l_best)
        with open('mripara0823.txt','a') as f:
            f.write(str(l_prec))
            f.write('\n')
            f.write(str(l_best))
            f.write('\n')


