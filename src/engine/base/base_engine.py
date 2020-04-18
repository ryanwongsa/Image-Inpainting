import torch
from torchvision import transforms
from torch.utils.data import DataLoader

try:
    from apex import amp
except:
    pass

from tqdm.auto import tqdm
import os
import cProfile
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint

class BaseEngine(object):
    def __init__(self, hparams):
        self.hparams = hparams
        
        self._init_configs()
        
        self._init_augmentation()
        self._init_train_datalader()
        self._init_valid_dataloader()
        
        self._init_model()
        self._init_optimizer()
        self._init_criterion_function()
        
        self._init_scheduler()
        
        self._init_metrics()
        self._init_logger()
        
        self.setup()
    
    def setup(self):
        self.trainer = Engine(self.train_step)
        self.evaluator = Engine(self.eval_step)
        
        self.logger._init_logger(self.trainer, self.evaluator)
        # TODO: Multi-gpu support
        self.model.to(self.device)
        if self.use_amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        
        if self.checkpoint_dir is not None:
            if not self.load_model_only:
                objects_to_checkpoint = {
                    "trainer": self.trainer,
                    "model": self.model, 
                    "optimizer": self.optimizer,
                    "scheduler": self.scheduler
                }
                if self.use_amp:
                    objects_to_checkpoint["amp"] = amp
            else:
                objects_to_checkpoint = {"model": self.model}
            objects_to_checkpoint = {k: v for k, v in objects_to_checkpoint.items() if v is not None}
            checkpoint = torch.load(self.checkpoint_dir)
            Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

        train_handler_params = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "metrics": self.train_metrics,
            "add_pbar": self.add_pbar
        }
        
        self.logger._add_train_events(**train_handler_params)
        
        
        to_save = {
                "model": self.model,
                "trainer": self.trainer,
                "optimizer": self.optimizer, 
                "scheduler": self.scheduler
            }
        if self.use_amp:
            to_save["amp"] = amp

        eval_handler_params = {
            "metrics": self.validation_metrics,
            "validloader": self.val_loader,
            "to_save": to_save,
            "add_pbar": self.add_pbar
        }
        
        eval_handler_params["to_save"] = {k: v for k, v in eval_handler_params["to_save"].items() if v is not None}
        self.logger._add_eval_events(**eval_handler_params)
        
        if self.scheduler:
            self.trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler)
    
    def train(self, run_params):
        self.trainer.run(self.train_loader,**run_params)
    
    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch, is_training = True)
        y_pred = self.model(x)
        loss, dict_losses = self.loss_fn(y_pred, y)
        self.loss_backpass(loss)
        self.optimizer.step()
        return self.output_transform(x, y, y_pred, loss, dict_losses)
    
    def loss_backpass(self, loss):
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    
    def eval_step(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = self.prepare_batch(batch, is_training = False)
            y_pred = self.model(x)
            loss, dict_losses = self.loss_fn(y_pred, y)
            return self.output_transform(x, y, y_pred, loss, dict_losses)
    
    def prepare_batch(self, batch, is_training=False):
        return batch
    
    def output_transform(self, x, y, y_pred, loss=None):
        return x, y, y_pred, loss
    
    def _init_scheduler(self):
        self.scheduler = None
    
    def get_batch(self):
        return next(iter(self.train_loader))
    
    def loss_fn(self, y_pred, y):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError
        
    def _init_optimizer(self):
        raise NotImplementedError

    def _init_criterion_function(self):
        raise NotImplementedError
        
    def _init_logger(self):
        raise NotImplementedError
        
    def _init_metrics(self):
        raise NotImplementedError
    
    def _init_train_datalader(self):
        raise NotImplementedError
        
    def _init_valid_dataloader(self):
        raise NotImplementedError
    
    def _init_augmentation(self):
        raise NotImplementedError 
    
    def _init_configs(self):
        raise NotImplementedError 