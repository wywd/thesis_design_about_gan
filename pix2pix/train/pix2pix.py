# training pix2pix using facades
# 导入包

import time
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

import model.models as model
from utils.image import ImageDataset

# 参数设置
start_epoch = 0  # 当前迭代次数
epochs = 100  # 最终迭代次数
decay_epoch = 100  # 减小学习速率的迭代轮次,取值范围：[start_epoch，epochs]
workers = 0  # 数据加载器进程数
batch_size = 1  # 批大小
image_size = 256  # 图像大小
channels = 3  # 图像通道数
lr = 0.0002  # 学习速率
n_gpu = 0  # GPU数量
beta1 = 0.5  # Adm优化器第一个超参数
beta2 = 0.999  # Adm优化器第二个超参数
L1_weight = 100  # L1 Loss权重
patch = (1, image_size // 2 ** 4, image_size // 2 ** 4)  # 输入判别器的patch大小

data_root = "../data/facades"  # 数据集路径
out_root = "../out"  # 输出路径（model/loss/images etc）

# 数据加载和预处理
transform = [
    transforms.Resize(256, Image.BICUBIC),  # 插值方式为双三次插值算法
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
train_loader = DataLoader(
    ImageDataset(data_root, transforms_=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
val_data_loader = DataLoader(
    ImageDataset(data_root, transforms_=transform, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=0
)

# 设备配置
device = torch.device(
    "cuda:0" if (
        torch.cuda.is_available() and n_gpu > 0) else "cpu")
if device.type == "cpu":
    print("----------使用CPU训练----------")
else:
    cudnn.benchmark = True  # 开启卷积优化
    print("----------使用GPU训练----------")

# 展示部分图像样本
# real_batch = next(iter(val_data_loader))
# imgA = real_batch['A']
# imgB = real_batch['B']
# img_sample = torch.cat((imgA.data, imgB.data), -2).squeeze(0)
# plt.figure()
# plt.axis("off")
# plt.title("Training Sample Images")
# plt.imshow(np.transpose(make_grid(img_sample, nrow=5, normalize=True), (1, 2, 0)))
# plt.savefig(out_root+'/img/training_sample_image.jpg')
# plt.show()

# 载入模型
netG = model.G(in_channels=channels, out_channels=channels).to(device)
if (device.type == 'cuda') and (n_gpu > 1):
    netG = nn.DataParallel(netG, list(range(n_gpu)))
netD = model.D(channels).to(device)
if (device.type == 'cuda') and (n_gpu > 1):
    netD = nn.DataParallel(netD, list(range(n_gpu)))

if start_epoch != 0:
    print("-----载入预训练模型继续训练-----")
    netG.load_state_dict(torch.load(out_root + "/load/pix2pix_G_{}.pth".format(start_epoch)))
    netD.load_state_dict(torch.load(out_root + "/load/pix2pix_D_{}.pth".format(start_epoch)))
else:
    print("-----训练新模型-----")
    netG.apply(model.weights_init_normal)  # 权重初始化
    netD.apply(model.weights_init_normal)

# 定义优化目标和损失函数
optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

criterion_GAN = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()


# 开始训练
G_losses = []
D_losses = []
print("开始训练...")
prev_time = time.time()
for epoch in range(start_epoch, epochs):
    for i, batch in enumerate(train_loader, 0):
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        real_labels = torch.ones(real_A.size(0), *patch, requires_grad=False).to(device)
        fake_labels = torch.zeros(real_A.size(0), *patch, requires_grad=False).to(device)
        # 训练鉴别器
        optimizer_D.zero_grad()
        d_out_real = netD(real_A, real_B)
        loss_real = criterion_GAN(d_out_real, real_labels)
        loss_real.backward()
        fake_A = netG(real_B)
        d_out_fake = netD(fake_A, real_B)
        loss_fake = criterion_GAN(d_out_fake, fake_labels)
        loss_fake.backward()
        loss_D = loss_real + loss_fake
        optimizer_D.step()
        # 训练生成器
        optimizer_G.zero_grad()
        fake_A = netG(real_B)
        g_d_out_fake = netD(fake_A, real_B)
        loss_GAN = criterion_GAN(g_d_out_fake, real_labels)
        loss_L1 = criterion_L1(fake_A, real_A)
        loss_G = loss_GAN + loss_L1 * L1_weight
        loss_G.backward()
        optimizer_G.step()

        # 性能评价|日志记录
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        if i % 1 == 0:
            time_left = time.time() - prev_time
            print(
                "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tL1: %.4f\tLoss_GAN: %.4f\t[spend : %.2fs]" %
                (epoch,
                 epochs,
                 i,
                 len(train_loader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_L1.item(),
                    loss_GAN.item(),
                    time_left))
    # 对于每一轮
    with torch.no_grad():
        images = next(iter(val_data_loader))
        imgA = images['A'].to(device)
        imgB = images['B'].to(device)
        fake = netG(imgB)
        img_sample = torch.cat((imgB.data, fake.data, imgA.data), -2).squeeze(0)
        plt.figure()
        plt.axis("off")
        plt.title("Training Sample Images")
        plt.imshow(
            np.transpose(
                make_grid( img_sample.detach().cpu(), nrow=5, normalize=True), (1, 2, 0)))
        plt.show()
        save_image(img_sample, out_root +"/img/training_sample_image{}.jpg".format(epoch), nrow=5, normalize=True)
# 损失函数
plt.figure(figsize=(10, 5))
plt.title("PixPix Training Loss")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("i")
plt.ylabel("loss")
plt.legend()
plt.savefig(out_root + "/evaluation/pix2pix_loss.png")
plt.show()
# 保存模型
torch.save(netG.state_dict(), out_root +
           "/load/pix2pix_G_{}.pth".format(epochs))
torch.save(netD.state_dict(), out_root +
           "/load/pix2pix_D_{}.pth".format(epochs))
