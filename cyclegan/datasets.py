import os
import glob
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        img1 = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        img2 = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        return {'A': img1, 'B': img2}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


if __name__ == '__main__':
    transform = [transforms.Resize(128, Image.BICUBIC),
                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    data_root = "./data/selfie2anime1200_2"
    data_loader = DataLoader(
        ImageDataset(data_root, transforms_=transform, mode="test"),
        batch_size=15,
        shuffle=True,
        num_workers=0
    )
    data = next(iter(data_loader))
    imgA = data["A"]
    imgB = data["B"]
    save_image(imgA, "test_sample_img.jpg", nrow=15, normalize=True)






