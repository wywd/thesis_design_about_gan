# original GAN model (Excepted dataset -> mnist)

import os
import torch
import torchvision
import torch.nn as nn


# model -> Discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
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
    def __init__(self, input_z):
        super(G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_z, 256),
            nn.ReLU(True),  # inplace = True,就地操作，节省内存
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()  # 输出数据分布于[-1, 1]
        )

    def forward(self, x):
        return self.model(x)

