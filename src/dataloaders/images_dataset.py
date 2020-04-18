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
    def __init__(self, root_dir, height, width, mask_generator, isValidation=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.height, self.width = height, width

        self.list_images = sorted(list(Path(self.root_dir).rglob('*.jpg')))
        self.mask_generator = mask_generator
        
        self.isValidation = isValidation
        self.length = len(self.list_images)
        
    def __len__(self):
        return self.length
    
    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)
    
    def __getitem__(self, idx):
        image_loc = self.list_images[idx]
        image = np.array(Image.open(image_loc).convert('RGB').resize((self.width, self.height)))
        
        if self.isValidation:
            mask = self.mask_generator.sample(index=idx, isValidation=self.isValidation)
        else:
            mask = self.mask_generator.sample()
            
        masked_img = np.copy(image)
        masked_img[mask==0.0] = 1.0
        
        if self.transform:
            masked_img = self.transform(masked_img)
            image = self.transform(image)
        
        return masked_img, torch.tensor(mask.astype('float')).permute(2,0,1), image