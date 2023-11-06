import ADmodel
import torch
import nibabel as nib
import numpy as np
import loader
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'save/f5/1_0.6475_0.8961_0.5963.pth'
img_path = 'ad_mci_pre/pre/AD_I260270.nii'

model = ADmodel.Model().cuda()
state = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(state['state_dict'], strict=True)
model.eval()

img = nib.load(img_path).get_fdata()
img = img[16:96, 19:119, 30:106]
img = np.squeeze(img)
img = img * 1.0
img = (img - img.min()) / (img.max() - img.min())
img = torch.from_numpy(img)
img = img.unsqueeze(0).float()
img = img.unsqueeze(0).float()  # batch维度
img = img.cuda()



with torch.no_grad():
    _, output = model(img)

pred = F.softmax(output, dim=1)

print(pred)

_,pred = torch.max(output, 1)
print(pred)

result = {0: 'AD', 1: 'MCI', 2: 'CN'}
print(f'预测结果为: {result[pred.item()]}')

