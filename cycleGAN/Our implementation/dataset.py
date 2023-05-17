from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class ObjectToObjectDataset(Dataset):
    def __init__(self, root_obj1, root_obj2, transform=None):
        self.root_obj1 = root_obj1
        self.root_obj2 = root_obj2
        self.transform = transform

        self.obj1_images = os.listdir(root_obj1)
        self.obj2_images = os.listdir(root_obj2)
        self.length_dataset = max(len(self.obj1_images), len(self.obj2_images))
        self.obj1_len = len(self.obj1_images)
        self.obj2_len = len(self.obj2_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        obj1_img = self.obj1_images[index % self.obj1_len]
        obj2_img = self.obj2_images[index % self.obj2_len]

        obj1_path = os.path.join(self.root_obj1, obj1_img)
        obj2_path = os.path.join(self.root_obj2, obj2_img)

        obj1_img = np.array(Image.open(obj1_path).convert("RGB"))
        obj2_img = np.array(Image.open(obj2_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=obj1_img, image0=obj2_img)
            obj1_img = augmentations["image"]
            obj2_img = augmentations["image0"]

        return obj1_img, obj2_img