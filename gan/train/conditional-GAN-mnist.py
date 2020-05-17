# training conditional-GAN using mnist
# 导入包

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import mnist
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import model.cgan as cgan
from utils.image import to_img

# 参数设置
epochs = 200  # 迭代次数
workers = 0  # 数据加载器进程数
batch_size = 64  # 批大小
image_size = 28  # 图像大小
num_classes = 10  # 图像类别数目
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
train_set = mnist.MNIST(data_root, train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 设备配置
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
if device.type == "cpu":
    print("----------使用CPU训练----------")
else:
    print("----------使用GPU训练----------")

# 载入模型
netG = cgan.G(z_dimension, num_classes).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netD = cgan.D(image_size**2, num_classes).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 定义优化目标和损失函数
criterion = nn.BCELoss()
d_optimizer = optim.Adam(netD.parameters(), lr=lr)
g_optimizer = optim.Adam(netG.parameters(), lr=lr)

# 开始训练

# 保存D和G的迭代损失值
d_model_losses = []
g_model_losses = []
# 构造固定的生成器输入作为评价基准
one_hot_labels = torch.zeros(10)
fixed_noise = torch.randn(100, z_dimension)  # 固定噪声
fixed_label = (torch.arange(100) % 10).view(-1, 1)
fixed_one_hot = torch.zeros(100, 10).scatter_(1, fixed_label, 1)  # 固定one-hot编码
fixed_data = torch.cat((fixed_noise, fixed_one_hot), 1)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        num_img = images.size(0)  # 当前输入的图片batch大小
        real_labels = torch.ones(num_img, 1).to(device)
        fake_labels = torch.zeros(num_img, 1).to(device)
        labels = labels.view(-1, 1)  # labels大小为(num_img×1)
        one_hot = torch.zeros(num_img, 10).scatter_(1, labels, 1)  # 将类标签转换为one-hot编码
        # 训练鉴别器
        d_optimizer.zero_grad()
        images = images.view(num_img, -1)  # 展开为:(num_img*784)
        d_input_real = torch.cat((images, one_hot), 1).to(device)
        d_out_real = netD(d_input_real)
        D_x = d_out_real.mean().item()
        d_loss_real = criterion(d_out_real, real_labels)
        d_loss_real.backward()
        noises = torch.randn(num_img, z_dimension)  # 随机生成噪声
        g_input = torch.cat((noises, one_hot), 1).to(device)  # 输入生成器G的随机噪声+标签约束
        g_out_fake = netG(g_input)  # 生成器生成的数据
        d_input_fake = torch.cat((g_out_fake, one_hot), 1)
        d_out_fake = netD(d_input_fake)
        D_G_z1 = d_out_fake.mean().item()
        d_loss_fake = criterion(d_out_fake, fake_labels)
        d_loss_fake.backward()
        d_loss = d_loss_real + d_loss_fake  # 鉴别器损失函数
        d_optimizer.step()
        # 训练生成器
        g_optimizer.zero_grad()
        g_out_fake_2 = netG(g_input)
        d_input_fake_2 = torch.cat((g_out_fake_2, one_hot), 1)
        d_out_fake_2 = netD(d_input_fake_2)
        D_G_z2 = d_out_fake_2.mean().item()
        g_loss = criterion(d_out_fake_2, real_labels)
        g_loss.backward()
        g_optimizer.step()
        if i % 50 == 0:
            print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                  % (epoch, epochs, i, len(train_loader), d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

        d_model_losses.append(d_loss.item())
        g_model_losses.append(g_loss.item())

    # 生成器性能评估
    fake_images = netG(fixed_data)
    fake_images = to_img(fake_images)
    save_image(fake_images, out_root+'/img/cgan_fake_images-{}.png'.format(epoch + 1), nrow=10)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.title("Conditional GAN Training loss")
plt.plot(g_model_losses, label="G")
plt.plot(d_model_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(out_root+'/evaluation/cgan_loss.jpg')
plt.show()

# 保存生成器模型
torch.save(netG.state_dict(), out_root+'load/cgan_generator.pth')









