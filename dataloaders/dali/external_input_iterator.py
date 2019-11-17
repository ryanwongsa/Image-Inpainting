import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import random
from pathlib import Path

class ExternalInputIterator(object):
    def __init__(self, image_dir, mask_dir, batch_size):
        self.images_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_files = list(Path(self.images_dir).rglob('*.jpg'))
        self.mask_files = list(Path(self.mask_dir).rglob('*.png'))
        shuffle(self.image_files)
        shuffle(self.mask_files)

        self.data_set_len = len(self.image_files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.image_files)
        return self

    def __next__(self):
        batch_imgs = []
        batch_masks = []
        for _ in range(self.batch_size):
            jpeg_img_filename = self.image_files[self.i]
            jpeg_mask_filename = random.choice(self.mask_files)
            f_img = open(jpeg_img_filename, 'rb')
            batch_imgs.append(np.frombuffer(f_img.read(), dtype = np.uint8))

            f_mask = open(jpeg_mask_filename, 'rb')
            batch_masks.append(np.frombuffer(f_mask.read(), dtype = np.uint8))

            self.i = (self.i + 1) % self.n
        return (batch_imgs, batch_masks)

    def __len__(self):
        return self.data_set_len

    @property
    def size(self,):
        return self.data_set_len

    next = __next__