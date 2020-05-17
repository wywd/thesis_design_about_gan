# 导入包
import torch
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

import models as model
from datasets import ImageDataset

# GPU数量
ngpu = 0

# 图片大小
image_size = 256

# 开启卷积优化
cudnn.benchmark = True

# 设置设备
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 路径
load_root = "./out/load/cycleGAN_G_A2B_125.pth"
data_root = "./data/selfie2anime"  # 数据集路径

# 加载模型
net = model.G(3, 3).to(device)

# net = nn.DataParallel(net)
net.load_state_dict(torch.load(load_root, map_location='cpu'))


if (device.type == 'cuda') and (ngpu > 1):
    net = nn.DataParallel(net, list(range(ngpu)))

# 数据加载和预处理
transform_ = [transforms.Resize(int(image_size*1.12), Image.BICUBIC),
              transforms.RandomCrop(image_size),
              transforms.RandomHorizontalFlip(),  # 随机水平翻转
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

train_loader = DataLoader(
    ImageDataset(data_root, transforms_=transform_, mode="test"),
    batch_size=10,
    shuffle=True,
    num_workers=0
)

if __name__ == '__main__':
    print(len(train_loader))
    for i, batch in enumerate(train_loader, 0):
        imgA = batch["A"].to(device)
        fake_B = net(imgA)
        img_sample = torch.cat((imgA.data, fake_B.data), -2).squeeze(0)
        plt.figure()
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(img_sample.detach().cpu(), nrow=10, normalize=True), (1, 2, 0)))
        plt.show()

