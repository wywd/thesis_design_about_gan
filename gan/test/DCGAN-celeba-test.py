# test for DCGAN with celeba

# 导入包
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import model.dcgan as dcgan

# GPU数量
ngpu = 0

# 开启卷积优化
cudnn.benchmark = True

# 设置设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 路径
load_root = "../out/load/G_dcgan_celeba.pth"

# 噪声维度
nz = 100

# 加载模型
model = dcgan.G(nz).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_root, map_location='cpu'))

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(model, list(range(ngpu)))

img_list = []
if __name__ == '__main__':
    for i in range(10):
        noise = torch.randn(64, nz, 1, 1, device=device)
        fake = model(noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title("Fake Images-{}".format(i))
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()


