from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from utils.util import TwoCropTransform


def data_loader(dataset="cifar10", batch_size=512, semi=False, semi_percent=10):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == 'cifar10h':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)

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
    sampler = None
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
    if semi:
        per = semi_percent / 100
        x = int(per * len(train_dataset))
        y = int(len(train_dataset) - x)
        train, _ = random_split(train_dataset, [x, y])
    else:
        train = train_dataset

    train, val = random_split(train, [int(0.8 * len(train)),
                                      len(train) - int(0.8 * len(train))], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              drop_last=False
                              )
    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16,
                            drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=16,
                             drop_last=False)

    targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
    return train_loader, val_loader, test_loader, targets


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
