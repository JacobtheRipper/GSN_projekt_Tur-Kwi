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
        
        str_label = self.annotation.iloc[index, 1]
        # Convert string labels to integers and then tensors
        label = tensor(string_to_int_label(str_label))

        if self.transform:
            image = self.transform(image)

        return (image, label)
    
    # Convert string labels to integers
    def string_to_int_label(str):
        if str == "Electronic":
            return 0
        if str == "Experimental":
            return 1
        if str == "Folk":
            return 2
        if str == "Hip-Hop":
            return 3
        if str == "Instrumental":
            return 4
        if str == "International":
            return 5
        if str == "Pop":
            return 6
        if str == "Rock":
            return 7
