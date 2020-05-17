# training original-GAN using celeba
# 导入包

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import model.dcgan as dcgan


# 参数设置
manualSeed = 999  # 设置一个随机种子，确保可重复性
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

epochs = 5  # 迭代次数
workers = 0  # 数据加载器进程数
batch_size = 128  # 批大小
image_size = 64  # 图像大小
lr = 0.0002  # 学习速率
z_dimension = 100  # 噪声维度
kd = 1  # 每轮迭代中，鉴别器优化次数
kg = 1  # 每轮迭代中，生成器优化次数
ngpu = 0  # GPU数量
beta1 = 0.5  # Adm优化器超参数

data_root = "../data/celeba"  # 数据集路径
out_root = "../out"  # 输出路径（model/loss/images etc）

cudnn.benchmark = True  # 开启卷积优化

# 数据加载和预处理
dataset = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# 设备配置
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
if device.type == "cpu":
    print("----------使用CPU训练----------")
else:
    print("----------使用GPU训练----------")

# 展示训练集部分图像
real_batch = next(iter(data_loader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('training_image.jpg')
plt.show()

# 载入模型
netG = dcgan.G(z_dimension).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netD = dcgan.D().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netG.apply(dcgan.weights_init)  # 权重初始化
netD.apply(dcgan.weights_init)

# 定义优化目标和损失函数
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# 开始训练
fixed_noise = torch.randn(64, z_dimension, 1, 1, device=device)
real_label = 1
fake_label = 0
img_list = []
G_losses = []
D_losses = []
iters = 0
print("开始训练...")
for epoch in range(epochs):
    for i, data in enumerate(data_loader, 0):
        # 训练鉴别器
        for _ in range(kd):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            noise = torch.randn(b_size, z_dimension, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
        # 训练生成器
        for _ in range(kg):
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(data_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

# 绘制Loss曲线
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.jpg')
plt.show()

# 显示真实图像
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# 显示最后一轮迭代的生成图像
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('real_and_fake.jpg')
plt.show()

# 保存生成器模型参数
torch.save(netD.state_dict(), out_root + "/load/G_dcgan_celeba.pth")
