from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms


def download(opt):
    #url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
    url ="https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
    download_and_extract_archive(url, download_root=opt.data_dir, filename="cifa100-c.tar")


class Data_un(Dataset):
    def __init__(self, img_path, tr="brightness.npy", severity=1, transform=transforms.ToTensor()):
        self.s = severity
        self.transform = transform
        self.path = os.path.join(img_path, tr)
        self.wh_images = np.load(self.path)
        self.images = self.wh_images[(10000 * (self.s - 1)):(10000 * self.s), ...]
        self.target_path = os.path.join(img_path, "labels.npy")
        self.targets = np.load(self.target_path)[0:10000]

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        img = img.permute(0, 2, 1)

        return img, np.int(target)

    def __len__(self):
        return len(self.targets)
