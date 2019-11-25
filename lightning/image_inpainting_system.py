import numpy as np
import os
import torch
import pytorch_lightning as pl

from dataloaders.dali import ExternalInputIterator, ExternalSourcePipeline, LightningDaliDataloader
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

from models.pconv_unet import PConvUNet
from models.vgg16_extractor import VGG16Extractor

from loss.loss_compute import LossCompute

from argparse import ArgumentParser
from utils import unnormalize

class ImageInpaintingSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(ImageInpaintingSystem, self).__init__()
        self.hparams = hparams
        
        self.loss_factors = {
            "loss_hole": hparams.loss_factor_hole, 
            "loss_valid": hparams.loss_factor_valid,
            "loss_perceptual": hparams.loss_factor_perceptual,
            "loss_style_out": hparams.loss_factor_out,
            "loss_style_comp": hparams.loss_factor_comp,
            "loss_tv": hparams.loss_factor_tv,
        }
        
        self.pConvUNet = PConvUNet()
        
        self.vgg16extractor = VGG16Extractor().to("cuda")
        for param in self.vgg16extractor.parameters():
            param.requires_grad = False
        self.lossCompute = LossCompute(self.vgg16extractor, self.loss_factors, device="cuda")

    def forward(self, masked_img_tensor, mask_tensor):
        return self.pConvUNet(masked_img_tensor, mask_tensor)

    def training_step(self, batch, batch_nb):
        images = batch[0]["images"]
        masks = batch[0]["masks"]
        masked_images = masks*images

        ls_fn = self.lossCompute.loss_total(masks)
        output = self.forward(masked_images, masks)
        loss, dict_losses = ls_fn(images, output)

        dict_losses_train = {}
        for key, value in dict_losses.items():
            dict_losses_train[key] = value.item()

        self.logger.experiment.add_scalars('loss/train',dict_losses_train, self.global_step)
        self.logger.experiment.add_scalars('loss/overview',{'train_loss': loss}, self.global_step)
        
        return {'loss': loss,'progress_bar': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        
        images = batch[0]["images"]
        masks = batch[0]["masks"]
        masked_images = masks*images
    
        ls_fn = self.lossCompute.loss_total(masks)
        output = self.forward(masked_images, masks)
        loss, dict_losses = ls_fn(images, output)

        psnr = self.lossCompute.PSNR(images, output)
        if batch_nb == 0:
            res = np.clip(unnormalize(output.detach().cpu().numpy()),0,1)
            original_img = np.clip(unnormalize(masked_images.detach().cpu().numpy()),0,1)
            target_img = np.clip(unnormalize(images.detach().cpu().numpy()),0,1)
            combined_imgs = []
            for i in range(images.shape[0]):
                combined_img = np.concatenate((original_img[i], res[i], target_img[i]), axis=1)
                combined_imgs.append(combined_img)
            combined_imgs = np.concatenate(combined_imgs)
            self.logger.experiment.add_image('images', combined_imgs, dataformats='HWC') 
        dict_valid = {'val_loss': loss.mean(), 'psnr': psnr.mean(), **dict_losses}
        
        return dict_valid
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        
        avg_loss_hole = torch.stack([x['loss_hole'] for x in outputs]).mean()
        avg_loss_valid = torch.stack([x['loss_valid'] for x in outputs]).mean()
        avg_loss_perceptual = torch.stack([x['loss_perceptual'] for x in outputs]).mean()
        avg_loss_style_out = torch.stack([x['loss_style_out'] for x in outputs]).mean()
        avg_loss_style_comp = torch.stack([x['loss_style_comp'] for x in outputs]).mean()
        avg_loss_tv = torch.stack([x['loss_tv'] for x in outputs]).mean()
        valid_dict = {
            "loss_hole": avg_loss_hole, 
            "loss_valid": avg_loss_valid,
            "loss_perceptual": avg_loss_perceptual,
            "loss_style_out": avg_loss_style_out,
            "loss_style_comp": avg_loss_style_comp,
            "loss_tv": avg_loss_tv
        }

        self.logger.experiment.add_scalars('loss/valid',valid_dict, self.global_step)
        self.logger.experiment.add_scalars('loss/overview',{'valid_loss': avg_loss}, self.global_step)

        tqdm_dict = {'valid_psnr': avg_psnr, 'val_loss': avg_loss}

        self.val_dataloader()[0].reset()
        return {'val_loss':avg_loss, 'progress_bar': tqdm_dict}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        eii = ExternalInputIterator(self.hparams.train_dir, self.hparams.mask_dir, self.hparams.batch_size, isShuffle=True)
        pipe = ExternalSourcePipeline(batch_size=self.hparams.batch_size, num_threads=self.hparams.num_workers, device_id = 0,
            external_data = eii, exec_async=False, exec_pipelined=False)
        dataloader = LightningDaliDataloader(pipe, eii.size, self.hparams.batch_size, last_batch_padded=False, fill_last_batch=True, output_map=["images", "masks"])
        return dataloader
    
    def on_epoch_end(self):
        self.train_dataloader().reset()

    @pl.data_loader
    def val_dataloader(self):
        eii = ExternalInputIterator(self.hparams.valid_dir, self.hparams.mask_dir, self.hparams.batch_size, isShuffle=False)
        pipe = ExternalSourcePipeline(batch_size=self.hparams.batch_size, num_threads=self.hparams.num_workers, device_id = 0,
            external_data = eii, exec_async=False, exec_pipelined=False)
        dataloader = LightningDaliDataloader(pipe, eii.size, self.hparams.batch_size, last_batch_padded=True, fill_last_batch=True, output_map=["images", "masks"])
        return dataloader
    
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--learning_rate', type=float, default=0.0002) 
        parser.add_argument('--batch_size', default=6, type=int)
        parser.add_argument('--num_workers', default=2, type=int)

        parser.add_argument('--train_dir', type=str)
        parser.add_argument('--valid_dir', type=str)
        parser.add_argument('--mask_dir', type=str)
        
        parser.add_argument('--loss_factor_hole', type=float, default=6.0)
        parser.add_argument('--loss_factor_valid', type=float, default=1.0)
        parser.add_argument('--loss_factor_perceptual', type=float, default=0.05)
        parser.add_argument('--loss_factor_out', type=float, default=120.0)
        parser.add_argument('--loss_factor_comp', type=float, default=120.0)
        parser.add_argument('--loss_factor_tv', type=float, default=0.1)
        return parser