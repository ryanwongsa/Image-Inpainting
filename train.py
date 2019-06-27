from dataloaders.mask_generator import MaskGenerator
from models.vgg16_extractor import VGG16Extractor
from models.pconv_unet import PConvUNet
from dataloaders.images_dataset import ImagesDataset
from loss.loss_compute import LossCompute

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import numpy as np
from PIL import Image
import os
from utils.helpers import make_dir
from utils.preprocessing import Preprocessor
import torch

import sys

DEVICE = "cuda"
HEIGHT, WIDTH = 512, 512
LR = 0.0002

preprocess = Preprocessor(DEVICE)

SAVED_PATH = sys.argv[1] #"saved_models/2019-06-12-22-06/" 

mask_generator = MaskGenerator("dataset/irregular_mask/disocclusion_img_mask/", HEIGHT,WIDTH, invert_mask=False) 
pConvUNet = PConvUNet()
pConvUNet.load_state_dict(torch.load(SAVED_PATH+"pconvunet.pth"))
pConvUNet = pConvUNet.to(DEVICE)
optimizer = optim.Adam(pConvUNet.parameters(), lr=LR)
vgg16extractor = VGG16Extractor().to(DEVICE)
lossCompute = LossCompute(vgg16extractor, device=DEVICE)

dataset = ImagesDataset("dataset/train",HEIGHT, WIDTH, mask_generator)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2)

date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
save_dir = "saved_models/"+date_time+"/"
samples_dir = "saved_models/"+date_time+"/samples/"

make_dir(save_dir)
make_dir(samples_dir)

pbar = tqdm(dataloader)
counter = 0
pConvUNet.train()

for masked_img, mask, image in pbar:
    
    img_tensor = preprocess.normalize(image.type(torch.float).to(DEVICE))
    mask_tensor = mask.type(torch.float).transpose(1, 3).to(DEVICE)
    masked_img_tensor = preprocess.normalize(masked_img.type(torch.float).to(DEVICE))

    optimizer.zero_grad()
    ls_fn = lossCompute.loss_total(mask_tensor)

    output = pConvUNet(masked_img_tensor, mask_tensor)
    loss = ls_fn(img_tensor, output)
    total_loss = loss.mean()
    psnr = lossCompute.PSNR(img_tensor,output)

    pbar.set_description(str(total_loss.detach().cpu().numpy()))
    
    total_loss.backward()
    optimizer.step()
    
    if counter%100 == 0:
        torch.save(pConvUNet.state_dict(), save_dir+"pconvunet.pth")
        pConvUNet.eval()
        res = pConvUNet(masked_img_tensor,mask_tensor)
        res = np.clip(preprocess.unnormalize(res).detach().cpu().numpy(),0,1)
        original_img = np.clip(preprocess.unnormalize(masked_img_tensor).detach().cpu().numpy(),0,1)
        combined_img = np.concatenate((original_img[0], res[0]))
        saveImg = Image.fromarray((combined_img*255).astype(np.uint8))
        imgname = samples_dir+"sample"+"{:08d}".format(counter)+".jpg"
        saveImg.save(imgname)
        pConvUNet.train()
    counter+=1