from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np


# Create dataset from ISIC images
class ISICDataset(Dataset):
    def __init__(self, img_path, transform=None, csv_path=None, test=False):
        if csv_path is not None:
            self.targets = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,
                                f'{self.targets.iloc[index, 0]}.jpg')
        img = Image.open(img_name)
        img = self.transform(img)
        if not self.test:
            labels = self.targets.iloc[index, 1:]
            labels = np.array([labels])
            labels = (labels != 0).argmax(axis=1)
            labels = labels.squeeze()
            #targets = targets.astype('float').reshape(-1, 7)
            return img, labels
        else:
            return {'image': img}

    def __len__(self):
        return len(self.targets)
