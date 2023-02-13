import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import LABEL_TO_ID


class ImageDataset(Dataset):
    def __init__(
            self, path, transforms=None, channels=None, left_side=None, right_side=None
    ):
        self.path = path
        self.filenames = os.listdir(path)
        self.transforms = transforms
        self.left_side = left_side
        self.right_side = right_side
        self.channels = channels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.path, filename)
        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)

            if self.channels == "mean":
                image = image.mean(axis=0, keepdims=True)
            elif self.channels == 3:
                image = image[0:3, :]

        if self.left_side and self.right_side:
            image = image[:, :, self.left_side: self.right_side + 1]

        label = filename[:-4]
        label_list = torch.tensor([LABEL_TO_ID[l] for l in label])
        return image, label_list
