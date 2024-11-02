import torch
import numpy as np
from torchvision import datasets
from torch.utils import data
from data.Augmentations import build_transform


class dataset_pairs(torch.utils.data.Dataset):
    def __init__(self, dataset, neighbor):
        self.dataset = dataset
        self.neighbor = neighbor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        outs = self.dataset[idx]
        i = np.random.randint(0, 5)
        pairs = self.dataset[self.neighbor[idx, i]]
        return [outs, pairs]


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        self.transform = transform

    def __call__(self, x):
        if self.transform:
            transform_image = [self.base_transform[0](x) for i in range(self.n_views)]
        else:
            transform_image = self.base_transform[1](x)
        return transform_image


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_dataset(self, name, train_dataset=True):
        if name == 'cifar10':
            ## CIFAR10
            datasets_transform = build_transform(train_dataset, name)
            cifar10_train = datasets.CIFAR10(root=self.root_folder, train=True, transform=datasets_transform,
                                             download=True)
            cifar10_test = datasets.CIFAR10(root=self.root_folder, train=False, transform=datasets_transform,
                                            download=True)
            cifar10 = data.ConcatDataset([cifar10_train, cifar10_test])
        elif name == 'cifar100':
            ## CIFAR100
            datasets_transform = build_transform(train_dataset, name)
            cifar100_train = datasets.CIFAR100(root=self.root_folder, train=True, transform=datasets_transform,
                                               download=True)
            cifar100_test = datasets.CIFAR100(root=self.root_folder, train=False, transform=datasets_transform,
                                              download=True)
            cifar100 = data.ConcatDataset([cifar100_train, cifar100_test])
        elif name == 'stl10_pretrain':
            datasets_transform = build_transform('train', name)
            ## STL10
            stl10_train = datasets.STL10(root=self.root_folder, split='train', transform=datasets_transform, download=True)
            stl10_test = datasets.STL10(root=self.root_folder, split='test', transform=datasets_transform, download=True)
            stl10_pretrain = data.ConcatDataset([stl10_train, stl10_test])
        elif name == 'stl10':
            ## STL10
            datasets_transform = build_transform(train_dataset, name)
            stl10_train = datasets.STL10(root=self.root_folder, split='train', transform=datasets_transform,
                                         download=True)
            stl10_test = datasets.STL10(root=self.root_folder, split='test', transform=datasets_transform,
                                        download=True)
            stl10 = data.ConcatDataset([stl10_train, stl10_test])
        elif name == 'imagenet10':
            ## IMAGENET10
            datasets_transform = build_transform(train_dataset, name)
            imagenet10 = datasets.ImageFolder(root='./datasets/ImageNet-10',
                                              transform=datasets_transform)
        elif name == 'imagenet_dogs':
            ## IMAGENET-DOG
            datasets_transform = build_transform(train_dataset, name)
            imagenet_dogs = datasets.ImageFolder(
                root='./datasets/ImageNet-dogs',
                transform=datasets_transform)

        elif name == 'tiny_imagenet':
            ## TINY-IMAGENET
            datasets_transform = build_transform(train_dataset, name)
            tiny_imagenet = datasets.ImageFolder(
                root='./datasets/tiny-imagenet-200/tiny-imagenet-200/train',
                transform=datasets_transform)

        datasets_class_num = {'cifar10': 10,
                              'cifar100': 20,
                              'stl10': 10,
                              'stl10_pretrain': 10,
                              'imagenet10': 10,
                              'imagenet_dogs': 15,
                              'tiny_imagenet': 200
                              }
        train_datasets = {'cifar10': lambda: cifar10,
                          'cifar100': lambda: cifar100,
                          'stl10': lambda: stl10,
                          'stl10_pretrain': lambda: stl10_pretrain,
                          'imagenet10': lambda: imagenet10,
                          'imagenet_dogs': lambda: imagenet_dogs,
                          'tiny_imagenet': lambda: tiny_imagenet
                          }
        try:
            dataset_fn = train_datasets[name]
            class_num = datasets_class_num[name]
        except KeyError:
            raise KeyError(f"{name} is not a valid Dataset selection")
        else:
            return dataset_fn(), class_num
