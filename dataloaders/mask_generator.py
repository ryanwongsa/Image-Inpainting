import os
import cv2
import numpy as np

class MaskGenerator():
    def __init__(self, filepath, height, width, invert_mask=True, channels=3, random_seed=None):
        self.filepath = filepath
        self.height, self.width, self.channels  = height, width, channels
        self.invert_mask = invert_mask
    
        filenames = [f for f in os.listdir(self.filepath)]
        self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        
        if random_seed:
            np.random.seed(random_seed)

        print("{} masks found: {}".format(len(self.mask_files), self.filepath))  
        
    def _load_mask(self, rotation=True, dilation=True):
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))
        mask = cv2.resize(mask,(self.width,self.height))
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8) 
            mask = cv2.erode(mask, kernel, iterations=1)

        if self.invert_mask:
            return (np.invert(mask > 1)).astype(np.uint8)
        else:
            return (mask > 1).astype(np.uint8)
      
    def sample(self, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        return self._load_mask()