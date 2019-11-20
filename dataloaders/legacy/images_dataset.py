import torch
import torchvision
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

import os
import copy
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagesDataset(Dataset):
    def __init__(self, root_dir, height, width, mask_generator, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.height, self.width = height, width

        self.list_images = list(Path(self.root_dir).rglob('*.jpg'))
        self.mask_generator = mask_generator
                                
    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image_loc = self.list_images[idx]
        image = np.array(Image.open(image_loc).convert('RGB').resize((self.width, self.height))) / 255
        
        mask = self.mask_generator.sample()

        if self.transform:
            image = self.transform(image)
            
        masked_img = np.copy(image)
        masked_img[mask==0] = 1
        
        return masked_img, mask, image