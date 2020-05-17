import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img1 = img.crop((0, 0, w / 2, h))
        img2 = img.crop((w / 2, 0, w, h))
        if np.random.random() < 0.5:  # 随机一部分图片镜像，可以用transforms.RandomHorizontalFlip()代替
            img1 = Image.fromarray(np.array(img1)[:, ::-1, :], "RGB")
            img2 = Image.fromarray(np.array(img2)[:, ::-1, :], "RGB")
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return {"A": img1, "B": img2}

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    transforms_ = [
        transforms.Resize(256, Image.BICUBIC),  # 插值方式为双三次插值算法
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    data_root = "../data/facades"
    data_loader = DataLoader(
        ImageDataset(data_root, transforms_=transforms_),
        batch_size=10,
        shuffle=True,
        num_workers=0
    )
    data = next(iter(data_loader))
    imgA = data['A']
    imgB = data['B']
    img_sample = torch.cat((imgA.data, imgB.data), -2).squeeze(0)
    # 显示样本图像
    plt.figure()
    plt.imshow(np.transpose(make_grid(img_sample, nrow=5, normalize=True), (1, 2, 0)))
    plt.show()
    # 保存样本图像
    save_image(img_sample, "../out/image_dataset_test.jpg", nrow=5, normalize=True)




