from dataloaders import ImagesDataset, MaskGenerator
from models import VGG16Extractor, PConvUNet
from torch.utils.data import DataLoader
from loss.loss_compute import LossCompute

import torch
from tqdm.auto import tqdm
import os
import cProfile
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint
from engine.base.base_engine import BaseEngine

class InpaintingEngine(BaseEngine):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    def prepare_batch(self, batch, is_training=False):
        masked_imgs, masks, images = batch
        masked_imgs = masked_imgs.float().to(self.device)
        masks = masks.float().to(self.device)
        images = images.float().to(self.device)
        self.ls_fn = self.lossCompute.loss_total(masks)
        return (masked_imgs, masks), images
    
    def loss_fn(self, y_pred, y):
        loss, dict_losses = self.ls_fn(y, y_pred)
        return loss, dict_losses
    
    def output_transform(self, x, y, y_pred, loss=None, dict_losses=None):
        return {
            "loss": loss.item(), 
            "dict_losses":dict_losses,
            "mask_image": x[0],
            "mask": x[1],
            "predicted": y_pred,
            "target": y
        }
        
    def _init_optimizer(self):
        if self.optimizer_name == "baseline":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = None

    def _init_criterion_function(self):
        from models import VGG16Extractor
        self.vgg16extract = VGG16Extractor()
        self.vgg16extract.to(self.device)
        self.vgg16extract.eval()
        self.lossCompute = LossCompute(self.vgg16extract, self.loss_factors, device=self.device)
        
    def _init_scheduler(self):
        if self.scheduler_name == "baseline":
            self.scheduler = None
        else:
            self.scheduler = None
        
    def _init_logger(self):
        from logger.neptune_logger import MyNeptuneLogger
        self.logger = MyNeptuneLogger(**self.logger_params)
        
    def _init_metrics(self):
        from ignite.metrics import Loss, RunningAverage
        from metrics import PSNR_Metric, Loss_Metric
        
        self.train_metrics = {
            'train_avg_loss': RunningAverage(output_transform=lambda x: x["loss"], alpha=0.98)
        }
        
        self.validation_metrics = {
            'psnr': PSNR_Metric(output_transform=lambda x: (x["predicted"], x["target"])),
            'loss': Loss_Metric(self.lossCompute, output_transform=lambda x: (x["mask"], x["predicted"], x["target"]))
        }
    
    def _init_model(self):
        if self.model_name == "baseline":
            from models import PConvUNet
            self.model = PConvUNet()
        else:
            self.model = None
    
    def _init_augmentation(self):
        # self.aug_name
        if self.aug_name == "baseline":
            from augmentations import transforms
            self.tfms = transforms
        else:
            self.tfms = None
        
    def _init_train_datalader(self):
        mg = MaskGenerator(**self.train_mask_params)
        trainset = ImagesDataset(mask_generator=mg, transform=self.tfms["train"], **self.train_ds_params)
        train_params = {
            "dataset": trainset, 
            "num_workers": self.num_workers,
            "batch_size":self.train_batch_size, 
            "shuffle":True, 
            "pin_memory":True, 
            "drop_last":True,
            "worker_init_fn":trainset.init_workers_fn
        }
        self.train_loader = DataLoader(**train_params)
        
    def _init_valid_dataloader(self):
        mg = MaskGenerator(**self.valid_mask_params)
        valset = ImagesDataset(mask_generator=mg, transform=self.tfms["valid"], **self.valid_ds_params)
        val_params = {
            "dataset": valset, 
            "num_workers": self.num_workers,
            "batch_size":self.val_batch_size, 
            "shuffle":False, 
            "pin_memory":True, 
            "drop_last":False,
            "worker_init_fn":valset.init_workers_fn
        }
        self.val_loader = DataLoader(**val_params)

    def _init_configs(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = self.hparams.checkpoint_dir
        self.valid_mask_params = self.hparams.valid_mask_params
        self.valid_ds_params = self.hparams.valid_ds_params
        self.num_workers = self.hparams.num_workers
        self.train_batch_size = self.hparams.train_batch_size
        self.val_batch_size = self.hparams.val_batch_size
        self.train_ds_params = self.hparams.train_ds_params
        self.train_mask_params = self.hparams.train_mask_params
        self.logger_params = self.hparams.logger_params
        self.loss_factors = self.hparams.loss_factors
        self.lr = self.hparams.lr
        
        self.aug_name = self.hparams.aug_name
        self.model_name = self.hparams.model_name
        self.scheduler_name = self.hparams.scheduler_name
        self.optimizer_name = self.hparams.optimizer_name
        
        self.add_pbar = self.hparams.add_pbar
        self.use_amp = self.hparams.use_amp
        self.load_model_only = self.hparams.load_model_only