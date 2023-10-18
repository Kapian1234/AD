# 0 ad
# 1 mri
# 2 cn
import pandas as pd
from torch.utils.data.dataset import Dataset
import nibabel as nib
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import DataLoader

# 平衡数据集
def recreate(df):
    img_oversample0 = []
    img_oversample1 = []

    df_oversample0 = pd.DataFrame()
    for i in range(len(df)):
        if df.iloc[i][2] == 0:
            img_oversample0.append(df.iloc[i][1])
            img_oversample0.append(df.iloc[i][1])
    df_oversample0['path'] = img_oversample0
    df_oversample0['label'] = [0] * len(img_oversample0)

    df_oversample1 = pd.DataFrame()
    for i in range(len(df)):
        if df.iloc[i][2] == 1:
            img_oversample1.append(df.iloc[i][1])
            img_oversample1.append(df.iloc[i][1])
    df_oversample1['path'] = img_oversample1
    df_oversample1['label'] = [1] * len(img_oversample1)

    df = pd.concat([df, df_oversample0], axis=0, ignore_index=True)
    df = pd.concat([df, df_oversample1], axis=0, ignore_index=True)
    return df


class newADdataset(Dataset):
    def __init__(self, csv_path, phase):
        self.df = pd.read_csv(csv_path)
        if phase == 'train':
            self.df=recreate(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx][1]
        # print(img_path)
        label = self.df.iloc[idx][2]
        img = nib.load(img_path).get_fdata()

        # 图像尺寸压缩
        resize_transform = transforms.Resize((80, 100, 76))
        img = resize_transform(img)
        img = np.squeeze(img)

        img = img*1.0
        img = (img-img.min())/(img.max()-img.min())
        img = torch.from_numpy(img)
        # img = torch.Tensor(1*[img.tolist()])
        img = img.unsqueeze(0).float()

        # 返回图像chwd为1*80*100*76
        return img, label

    def __len__(self):
        return  len(self.df['label'])


def newADdataloader(phase, csv_path, batch_size):
    path = csv_path
    print(path)
    ADdataset = newADdataset(path, phase)

    if(phase == 'train'):
        dataloader = DataLoader(dataset=ADdataset, batch_size=batch_size, num_workers=0, shuffle=True)
    else:
        dataloader = DataLoader(dataset=ADdataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # print('dataloader')
    return dataloader
