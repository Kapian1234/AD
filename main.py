import model
import loader
import torch
import val


class Params:
    epochs = 20
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

    def start(self, i, type):


