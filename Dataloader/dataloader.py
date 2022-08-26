from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import torch

from utils.util import TwoCropTransform
from utils.isic import set_path
# from isicDataset import ISICDataset
import numpy as np


def generate_dataloader(data, name, transform=None):
    if data is None:
        return None
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Read image files to pytorch dataset using ImageFolder, a generic data
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 16}
    else:
        kwargs = {}

    # Wrap image dataset (defined above) in dataloader
    dataloader = DataLoader(dataset, batch_size=8,
                            shuffle=(name == "train"),
                            **kwargs)

    return dataloader


def data_loader(dataset="cifar10", batch_size=512, semi=False, semi_percent=10):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    elif dataset == "isic":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    # datasets
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='../../DATA2/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='../../DATA2/', train=False, download=True, transform=val_transform)

    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='../../DATA2/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='../../DATA2/', train=False, download=True, transform=val_transform)
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(
            root='.../../DATA2/', split="train", download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            root='../../DATA2/', split="test", download=True, transform=val_transform
        )
    elif dataset == "isic":
        train_img_path, train_class_path, val_img_path, val_class_path, test_img = set_path()
        train_dataset = ISICDataset(train_img_path, transform=train_transform,
                                    csv_path=train_class_path,
                                    test=False)
        test_dataset = ISICDataset(val_img_path, transform=train_transform,
                                   csv_path=val_class_path,
                                   test=False)
    if semi:
        per = semi_percent / 100
        x = int(per * len(train_dataset))
        y = int(len(train_dataset) - x)
        train, _ = random_split(train_dataset, [x, y])
    else:
        train = train_dataset

    train, val = random_split(train, [int(0.8 * len(train)),
                                      len(train) - int(0.8 * len(train))])

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              drop_last=False)
    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=16,
                            drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=16,
                             drop_last=False)

    targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
    return train_loader, val_loader, test_loader, targets


def tiny_imagenet_dataloader():
    train_loader = generate_dataloader("./datasets/tiny-imagenet-200/train", "train", transform=None)
    val_loader = generate_dataloader("./datasets/tiny-imagenet-200/val", "val", transform=None)
    test_loader = generate_dataloader("./datasets/tiny-imagenet-200/test", "test", transform=None)
    return train_loader, val_loader, test_loader


def set_loader(dataset, batch_size, num_workers, semi_percent=10, data_folder='../../DATA2/', pu=False, semi=False):
    # construct data loader
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    elif dataset == "isic":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if semi_percent == 10:
        x = 5000
        y = 45000
    elif semi_percent == 1:
        x = 500
        y = 49500

    if dataset == 'cifar10':
        if pu:
            # Downloading/Loading CIFAR10 data
            trainset = datasets.CIFAR10(root=data_folder,
                                        transform=train_transform,
                                        download=True)

            classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                         'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
            # Separating trainset/testset data/label
            x_train = trainset.data
            y_train = np.array(trainset.targets)
            subset_x_train = x_train[np.isin(np.array(y_train), [0, 1, 8, 9]).flatten()]
            subset_y_train = y_train[np.isin(np.array(y_train), [0, 1, 8, 9]).flatten()]
            trainset.data = subset_x_train
            trainset.targets = subset_y_train
            per = 1 / 100
            x = int(per * len(trainset))
            y = int(len(trainset) - x)
            train_dataset, _ = random_split(trainset, [x, y])
        else:
            train_dataset = datasets.CIFAR10(root=data_folder,
                                             transform=train_transform,
                                             download=True)
        val_dataset = datasets.CIFAR10(root=data_folder,
                                       train=False,
                                       transform=val_transform)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=data_folder,
                                        train=False,
                                        transform=val_transform)
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(
            root=data_folder, split="train", download=True, transform=train_transform
        )
        val_dataset = datasets.SVHN(
            root=data_folder, split="test", download=True, transform=val_transform
        )

    else:
        raise ValueError(dataset)
    if semi:
        per = semi_percent / 100
        x = int(per * len(train_dataset))
        y = int(len(train_dataset) - x)
        train, _ = random_split(train_dataset, [x, y])
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=False, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    return train_loader, val_loader


def set_loader_simclr(dataset, batch_size, num_workers, data_dir='../../DATA2/', size=32):
    # construct data loader
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    elif dataset == "isic":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(
            root=data_dir, split="train", download=True, transform=TwoCropTransform(train_transform)
        )
    else:
        raise ValueError(dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=train_sampler)

    return train_loader
