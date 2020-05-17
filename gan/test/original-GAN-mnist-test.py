# test for original GAN with mnist

# 导入包
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

import model.originalgan as originalgan
from utils.image import to_img

# GPU数量
ngpu = 0

# 开启卷积优化
cudnn.benchmark = True

# 设置设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 路径
load_root = "../out/load/G_gan_mnist.pth"

# 噪声维度
nz = 100

# 加载模型
model = originalgan.G(nz).to(device)
model.load_state_dict(torch.load(load_root, map_location='cpu'))

if __name__ == '__main__':
    for i in range(10):
        noise = torch.randn(64, nz).to(device)
        fake = model(noise)
        fake = to_img(fake)
        save_image(fake, "fake-{}.png".format(i))


