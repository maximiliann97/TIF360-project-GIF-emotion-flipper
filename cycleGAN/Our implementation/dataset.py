import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Object1Object2Dataset(Dataset):
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
            obj1_img = Image.fromarray(obj1_img)
            obj2_img = Image.fromarray(obj2_img)

            obj1_img = self.transform(obj1_img)
            obj2_img = self.transform(obj2_img)

        return obj1_img, obj2_img
