# training cycleGAN using selfie2anime
# 导入包

import time
import torch
import random
import itertools
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import models as model
import myoptim as myoptim
from datasets import ImageDataset


# 参数设置
start_epoch = 0  # 当前迭代次数
epochs = 200  # 最终迭代次数
decay_epoch = 100  # 减小学习速率的迭代轮次,取值范围：[start_epoch，epochs]
workers = 2  # 数据加载器进程数
batch_size = 1  # 批大小
image_size = 256  # 图像大小
channels = 3  # 图像通道数
lr = 0.0002  # 学习速率
n_gpu = 1  # GPU数量
beta1 = 0.5  # Adm优化器第一个超参数
beta2 = 0.999  # Adm优化器第二个超参数
L1_weight = 10.0  # 周期一致性损失权重(cycle consistency loss)
idt_weight = L1_weight * 0.5  # 标识映射损失权重(identity mapping loss)

data_root = "./data/selfie2anime"  # 数据集路径
out_root = "./out"  # 输出路径（model/loss/images etc）

# 数据加载和预处理
transform_ = [transforms.Resize(image_size, Image.BICUBIC),
              transforms.RandomHorizontalFlip(),  # 随机水平翻转
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

train_loader = DataLoader(
    ImageDataset(data_root, transforms_=transform_, mode="train"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True
)

data_loader = DataLoader(
    ImageDataset(data_root, transforms_=transform_, mode="test"),
    batch_size=10,
    shuffle=True,
    num_workers=0
)

# 设备配置
device = torch.device(
    "cuda:0" if (
        torch.cuda.is_available() and n_gpu > 0) else "cpu")
if device.type == "cpu":
    print("使用CPU训练...")
else:
    cudnn.benchmark = True  # 开启卷积优化
    print("使用GPU训练...(GPU = %d)" % n_gpu)
    
test_data = next(iter(data_loader))
test_imgA = test_data["A"].to(device)

# 载入模型
netG_A2B = model.G(input_nc=channels, output_nc=channels).to(device)
netG_B2A = model.G(input_nc=channels, output_nc=channels).to(device)
if (device.type == 'cuda') and (n_gpu > 1):
    netG_A2B = nn.DataParallel(netG_A2B, list(range(n_gpu)))
    netG_B2A = nn.DataParallel(netG_B2A, list(range(n_gpu)))
netD_A = model.D(channels).to(device)
netD_B = model.D(channels).to(device)
if (device.type == 'cuda') and (n_gpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(n_gpu)))
    netD_B = nn.DataParallel(netD_B, list(range(n_gpu)))

if start_epoch != 0:
    print("载入预训练模型继续训练...")
    netG_A2B.load_state_dict(torch.load(out_root + "/load/cycleGAN_G_A2B_{}.pth".format(start_epoch)))
    netG_B2A.load_state_dict(torch.load(out_root + "/load/cycleGAN_G_B2A_{}.pth".format(start_epoch)))
    netD_A.load_state_dict(torch.load(out_root + "/load/cycleGAN_D_A_{}.pth".format(start_epoch)))
    netD_B.load_state_dict(torch.load(out_root + "/load/cycleGAN_D_B_{}.pth".format(start_epoch)))
else:
    print("训练新模型...")
    netG_A2B.apply(model.weights_init_normal)  # 权重初始化
    netG_B2A.apply(model.weights_init_normal)
    netD_A.apply(model.weights_init_normal)
    netD_B.apply(model.weights_init_normal)

# 损失函数
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)

# 优化器
optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(beta1, beta2))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(beta1, beta2))

# LR调度器
lr_scheduler_G = optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=myoptim.LambdaLR(epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=myoptim.LambdaLR(epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=myoptim.LambdaLR(epochs, start_epoch, decay_epoch).step)

# 训练
G_losses = []  # 保存损失，用于绘制损失曲线
G_identity_losses = []
G_GAN_losses = []
G_cycle_losses = []
D_losses = []
fake_A_buffer = myoptim.ReplayBuffer()  # 使用生成图的历史更新鉴别器，防止模型抖动
fake_B_buffer = myoptim.ReplayBuffer()
print("开始训练...")
prev_time = time.time()
for epoch in range(start_epoch, epochs):
    for i, batch in enumerate(train_loader, 0):
        real_A = batch["A"].to(device)
        real_B = batch["B"].to(device)
        target_real = torch.ones(real_A.size(0), 1, requires_grad=False).to(device)
        target_fake = torch.zeros(real_A.size(0), 1, requires_grad=False).to(device)
        # 训练生成器 A2B and B2A
        # 1. Identity loss
        optimizer_G.zero_grad()
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * idt_weight
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * idt_weight
        # 2. GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        # 3. Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * L1_weight
        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * L1_weight
        # Total Loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()
        # 训练鉴别器 D_A
        optimizer_D_A.zero_grad()
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        # 训练鉴别器 D_B
        optimizer_D_B.zero_grad()
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        # 性能评价|日志记录
        if i % 60 == 0:
            time_left = time.time() - prev_time
            print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_G_identity: %.4f"
                  "\tLoss_G_GAN: %.4f\tLoss_G_cycle: %.4f\t[spend : %.2fs]" %
                  (epoch, epochs, i, len(train_loader), loss_D_A.item()+loss_D_B.item(), loss_G.item(),
                   loss_identity_A.item()+loss_identity_B.item(), loss_GAN_A2B.item()+loss_GAN_B2A.item(),
                   loss_cycle_ABA.item()+loss_cycle_BAB.item(),   time_left))
    # 对于每一个epoch
    G_losses.append(loss_G.item())
    G_identity_losses.append(loss_identity_A.item()+loss_identity_B.item())
    G_GAN_losses.append(loss_GAN_A2B.item()+loss_GAN_B2A.item())
    G_cycle_losses.append(loss_cycle_ABA.item()+loss_cycle_BAB.item())
    D_losses.append(loss_D_A.item()+loss_D_B.item())
    with torch.no_grad():
        fake_B = netG_A2B(test_imgA)
        img_sample = torch.cat((test_imgA.data, fake_B.data), -2).squeeze(0)
        save_image(img_sample, out_root + "/img/sample_{}.jpg".format(epoch), nrow=10, normalize=True)
    # 更新lr
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    if epoch > 99 and epoch % 10 == 0:
        # 保存模型
        torch.save(netG_A2B.state_dict(), out_root + "/load/cycleGAN_G_A2B_{}.pth".format(epoch))
        torch.save(netG_B2A.state_dict(), out_root + "/load/cycleGAN_G_B2A_{}.pth".format(epoch))
        torch.save(netD_A.state_dict(), out_root + "/load/cycleGAN_D_A_{}.pth".format(epoch))
        torch.save(netD_B.state_dict(), out_root + "/load/cycleGAN_D_B_{}.pth".format(epoch))
        
torch.save(netG_A2B.state_dict(), out_root + "/load/cycleGAN_G_A2B_{}.pth".format(epochs))
torch.save(netG_B2A.state_dict(), out_root + "/load/cycleGAN_G_B2A_{}.pth".format(epochs))
torch.save(netD_A.state_dict(), out_root + "/load/cycleGAN_D_A_{}.pth".format(epochs))
torch.save(netD_B.state_dict(), out_root + "/load/cycleGAN_D_B_{}.pth".format(epochs))

print("训练结束...")
# 绘制损失曲线
plt.figure()
plt.title("Loss_G")
plt.plot(G_losses)
plt.xlabel("epochs")
plt.ylabel("loss_G")
plt.savefig(out_root + "/evaluation/cycle_G_loss.jpg")

plt.figure()
plt.title("Loss_D")
plt.plot(D_losses)
plt.xlabel("epochs")
plt.ylabel("loss_D")
plt.savefig(out_root + "/evaluation/cycle_D_loss.jpg")

plt.figure()
plt.title("Loss_G_GAN")
plt.plot(G_GAN_losses)
plt.xlabel("i")
plt.ylabel("loss_G_GAN")
plt.savefig(out_root + "/evaluation/cycle_Loss_G_GAN.jpg")

plt.figure()
plt.title("Loss_G_identity")
plt.plot(G_identity_losses)
plt.xlabel("i")
plt.ylabel("Loss_G_identity")
plt.savefig(out_root + "/evaluation/cycle_Loss_G_identity.jpg")

plt.figure()
plt.title("Loss_G_cycle")
plt.plot(G_cycle_losses)
plt.xlabel("i")
plt.ylabel("loss_G_cycle")
plt.savefig(out_root + "/evaluation/cycle_Loss_G_cycle.jpg")





























