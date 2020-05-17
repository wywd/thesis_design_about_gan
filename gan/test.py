import torch

fixed_label = (torch.arange(100) % 10).view(-1, 1)
fixed_one_hot = torch.zeros(100, 10).scatter_(1, fixed_label, 1)  # 固定one-hot编码



