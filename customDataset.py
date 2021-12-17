import os
import pandas as pd  # CSV file parsing
from torch.utils.data import Dataset
from torch import tensor
from skimage import io

class customDataset(Dataset):
    def __init__(self, annotation_file, data_dir, transform=None):
        self.annotation = pd.read_csv(annotation_file)
        self.data_dir = data_dir
        self.transform = transform

    # returns the size of the dataset
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_dir = os.path.join(self.data_dir, self.annotation.iloc[index, 0])
        image = io.imread(img_dir)
        label = tensor(int(self.annotation.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, label)
