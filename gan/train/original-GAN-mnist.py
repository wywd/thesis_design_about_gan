# training original-GAN using mnist
# 导入包
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

import model.originalgan as originalgan
from utils.image import to_img

# 参数设置
epochs = 200  # 迭代次数
workers = 0  # 数据加载器进程数
batch_size = 128  # 批大小
image_size = 28  # 图像大小
lr = 0.0003  # 学习速率
z_dimension = 100  # 噪声维度
kd = 1  # 每轮迭代中，鉴别器优化次数
kg = 1  # 每轮迭代中，生成器优化次数
ngpu = 0  # GPU数量

data_root = "../data/mnist"  # 数据集路径
out_root = "../out"  # 输出路径（model/loss/images etc）

cudnn.benchmark = True  # 开启卷积优化

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_set = MNIST(data_root, train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)

# 设备配置
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
if device.type == "cpu":
    print("----------使用CPU训练----------")
else:
    print("----------使用GPU训练----------")


# 载入模型
netG = originalgan.G(input_z=z_dimension).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netD = originalgan.D().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 定义优化目标和损失函数
criterion = nn.BCELoss()
d_optimizer = optim.Adam(netD.parameters(), lr=lr)
g_optimizer = optim.Adam(netG.parameters(), lr=lr)


# 开始训练
D_losses = []
G_losses = []
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        num_img = images.size(0)  # 当前输入的图片batch大小
        images = images.view(num_img, -1).to(device)  # num_img*784
        real_labels = torch.ones(num_img, 1).to(device)
        fake_labels = torch.zeros(num_img, 1).to(device)
        noise = torch.randn(num_img, z_dimension).to(device)
        # 训练鉴别器
        for _ in range(kd):
            netD.zero_grad()
            output = netD(images)
            D_x = output.mean().item()
            errD_real = criterion(output, real_labels)
            # errD_real.backward()
            fake_images = netG(noise)
            output = netD(fake_images)
            D_G_z1 = output.mean().item()
            errD_fake = criterion(output, fake_labels)
            # errD_fake.backward()
            errD = errD_real + errD_fake
            errD.backward()
            d_optimizer.step()
        # 训练生成器
        for _ in range(kg):
            netG.zero_grad()
            output = netD(netG(noise))
            errG = criterion(output, real_labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            g_optimizer.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        D_losses.append(errD.item())
        G_losses.append(errG.item())

    if epoch == 0:
        real_images = to_img(images)
        save_image(real_images, out_root + "/img/real_image.png")

    fake_images = to_img(fake_images)
    save_image(fake_images, out_root + "/img/fake_images-{}.png".format(epoch + 1))

plt.figure(figsize=(10, 5))
plt.title("original GAN Training Loss")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("i")
plt.ylabel("loss")
plt.legend()
plt.savefig(out_root + "/evaluation/original_gan_loss.png")
plt.show()
# 保存生成器模型参数
torch.save(netG.state_dict(), out_root + "/load/G_gan_mnist.pth")



















