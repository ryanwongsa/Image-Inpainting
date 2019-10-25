import numpy as np
from PIL import Image
import os
import torch
import pytorch_lightning as pl

from dataloaders.mask_generator import MaskGenerator
from dataloaders.images_dataset import ImagesDataset
from torch.utils.data import DataLoader

from models.pconv_unet import PConvUNet
from models.vgg16_extractor import VGG16Extractor

from loss.loss_compute import LossCompute

from utils.preprocessing import Preprocessor

from argparse import ArgumentParser

class ImageInpaintingSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(ImageInpaintingSystem, self).__init__()
        self.hparams = hparams
        self.pConvUNet = PConvUNet()
        
        vgg16extractor = VGG16Extractor().to("cuda")
        for param in vgg16extractor.parameters():
            param.requires_grad = False
        self.lossCompute = LossCompute(vgg16extractor)
        
        self.preprocess = Preprocessor("cuda")

    def forward(self, masked_img_tensor, mask_tensor):
        return self.pConvUNet(masked_img_tensor, mask_tensor)

    def training_step(self, batch, batch_nb):
        masked_img, mask, image  = batch
        
        img_tensor = self.preprocess.normalize(image.type(torch.float))
        mask_tensor = mask.type(torch.float).transpose(1, 3)
        masked_img_tensor = self.preprocess.normalize(masked_img.type(torch.float))
        
        ls_fn = self.lossCompute.loss_total(mask_tensor)
        output = self.forward(masked_img_tensor, mask_tensor)
        loss = ls_fn(img_tensor, output).mean()
        
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        masked_img, mask, image  = batch
        
        img_tensor = self.preprocess.normalize(image.type(torch.float))
        mask_tensor = mask.type(torch.float).transpose(1, 3)
        masked_img_tensor = self.preprocess.normalize(masked_img.type(torch.float))
        
        ls_fn = self.lossCompute.loss_total(mask_tensor)
        output = self.forward(masked_img_tensor, mask_tensor)
        loss = ls_fn(img_tensor, output)
        
        psnr = self.lossCompute.PSNR(img_tensor, output)

        res = np.clip(self.preprocess.unnormalize(output).detach().cpu().numpy(),0,1)
        original_img = np.clip(self.preprocess.unnormalize(masked_img_tensor).detach().cpu().numpy(),0,1)
        combined_img = np.concatenate((original_img[0], res[0]))
        self.logger.experiment.add_image('images', combined_img, dataformats='HWC')   
        return {'val_loss': loss, 'psnr': psnr}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        tqdm_dict = {'valid_psnr': avg_psnr, 'valid_loss': avg_loss}
        return {'log':tqdm_dict,'valid_loss': avg_loss, 'valid_psnr': avg_psnr}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        mask_generator = MaskGenerator(self.hparams.mask_dir, self.hparams.height, self.hparams.width, invert_mask=self.hparams.invert_mask) 
        dataset = ImagesDataset(self.hparams.train_dir, self.hparams.height, self.hparams.width, mask_generator)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
        return dataloader
    
    @pl.data_loader
    def val_dataloader(self):
        mask_generator = MaskGenerator(self.hparams.mask_dir, self.hparams.height, self.hparams.width, invert_mask=self.hparams.invert_mask) 
        dataset = ImagesDataset(self.hparams.valid_dir, self.hparams.height, self.hparams.width, mask_generator)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
        return dataloader
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--height', default=256)
        parser.add_argument('--width', default=256)
        parser.add_argument('--learning_rate', default=0.0002) 
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--num_workers', default=2, type=int)

        
        parser.add_argument('--invert_mask', default=False, type=bool)

        parser.add_argument('--train_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--mask_dir', type=str)

        return parser