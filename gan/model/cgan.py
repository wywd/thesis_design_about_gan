# Conditional GAN model (Excepted dataset -> mnist)
# 导入包

import torch
import torch.nn as nn


# model -> Discriminator
class D(nn.Module):
    def __init__(self, num_img, num_classes):
        super(D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_img+num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# model -> Generator
class G(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim+num_classes, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()  # 使用Tanh，使生成的数据分布在[-1,1]之间
        )

    def forward(self, x):
        return self.model(x)



