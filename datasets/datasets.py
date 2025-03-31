import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST

import numpy as np
from PIL import Image
from utils import mkdir_if_missing


class CIFAR10Subset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = torch.tensor(img).permute(2, 0, 1) / 255.0  # 转换为 PyTorch 格式
        if self.transform:
            img = self.transform(img)
        return img, target

class CIFAR10(object):
    def __init__(self, **options):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = 64
        data_root = '/media/admin1/yhy/ARPL_muti/data/cifar10'

        pin_memory = True

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

        # 获取CIFAR-10的图像和标签数据
        full_data = trainset.data  # numpy array of shape (50000, 32, 32, 3)
        full_targets = trainset.targets  # list of length 50000

        # 创建筛选后的数据和标签列表
        filtered_data = []
        filtered_targets = []
        num_samples_per_class = 40

        # 按类别筛选
        for class_id in range(10):
            class_indices = [i for i, label in enumerate(full_targets) if label == class_id]
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
            filtered_data.extend(full_data[selected_indices])
            filtered_targets.extend([class_id] * num_samples_per_class)

        # 转换为 numpy 数组以创建自定义数据集
        filtered_data = np.array(filtered_data)
        filtered_targets = np.array(filtered_targets)

        filtered_trainset = CIFAR10Subset(filtered_data, filtered_targets, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(
            filtered_trainset, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader
        self.data = trainloader.dataset.data


class SVHN(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'svhn')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.trainloader = trainloader
        self.testloader = testloader


__factory = {
    'cifar10': CIFAR10,
    'svhn':SVHN,
}

def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)

