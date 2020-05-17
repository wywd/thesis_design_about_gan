# test for conditional GAN with mnist

# 导入包
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

import model.cgan as cgan
from utils.image import to_img

# GPU数量
ngpu = 0

# 开启卷积优化
cudnn.benchmark = True

# 设置设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 路径
load_root = "../out/load/G_cgan_mnist.pth"

# 噪声维度
nz = 100

# 图像类别数目
num_classes = 10

# 加载模型
model = cgan.G(nz, num_classes).to(device)
model.load_state_dict(torch.load(load_root, map_location='cpu'))


label = (torch.arange(100) % 10).view(-1, 1)
one_hot = torch.zeros(100, 10).scatter_(1, label, 1)
if __name__ == '__main__':
    for i in range(10):
        noise = torch.randn(100, nz)
        data = torch.cat((noise, one_hot), 1).to(device)
        fake = model(data)
        fake = to_img(fake)
        save_image(fake, "fake-{}.png".format(i), nrow=10)
