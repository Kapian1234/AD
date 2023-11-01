import ADmodel
import torch
import nibabel as nib
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'save/f1/0_0.717_0.0957_0.8978.pth'
img_path = 'cn_pre/pre/I225562.nii'

model = ADmodel.Model().cuda()
state = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
model.load_state_dict(state['state_dict'], strict=True)
model.eval()

img = nib.load(img_path).get_fdata()
img = img[40:120, 30:130, 10:86]
img = np.squeeze(img)
img = img * 1.0
img = (img - img.min()) / (img.max() - img.min())
img = torch.from_numpy(img)
img = img.unsqueeze(0).float()
img = img.cuda()

x_linear, output = model(img)

print(output)

