class Parameters(object):
    def __init__(self, hparams=None):
        
        self.aug_name = hparams.aug_name
        self.model_name = hparams.model_name
        self.scheduler_name = hparams.scheduler_name
        self.optimizer_name = hparams.optimizer_name
        
        self.epochs = hparams.epochs
        self.epoch_length = hparams.epoch_length
        
        self.train_ds_params = {
            "root_dir": hparams.train_images_dir,
            "height": hparams.height, 
            "width": hparams.width, 
        }
        
        self.train_mask_params = {
            "filepath": hparams.train_mask_dir,
            "height": hparams.height, 
            "width": hparams.width, 
            "invert_mask": hparams.train_invert_mask, 
            "channels": hparams.channels
        }
        
        self.valid_ds_params = {
            "root_dir": hparams.valid_images_dir,
            "height": hparams.height, 
            "width": hparams.width, 
            "isValidation": True,
        }
        
        self.valid_mask_params = {
            "filepath":  hparams.valid_mask_dir,
            "height": hparams.height, 
            "width": hparams.width, 
            "invert_mask": hparams.valid_invert_mask, 
            "channels": hparams.channels
        }

        self.num_workers = hparams.num_workers
        self.train_batch_size = hparams.train_bs
        self.val_batch_size = hparams.val_bs
        self.checkpoint_dir = hparams.checkpoint_dir
        self.lr = hparams.lr

        self.loss_factors = {
            "loss_hole": hparams.loss_hole, 
            "loss_valid": hparams.loss_valid,
            "loss_perceptual": hparams.loss_perceptual,
            "loss_style_out": hparams.loss_style_out,
            "loss_style_comp": hparams.loss_style_comp,
            "loss_tv": hparams.loss_tv
        }
        
        self.add_pbar = hparams.add_pbar
        self.use_amp = hparams.use_amp
        self.load_model_only = hparams.load_model_only
        
        self.logger_params = {
            "save_dir" : hparams.save_dir,
            "n_saved": hparams.n_saved,
            "prefix_name": hparams.name,
            "log_every": hparams.log_every,
            "project_name": "gatletag/image-inpainting",
            "name": hparams.name,
            "tags": [item for item in hparams.tags.split(',')],
            "params": {
                **self.loss_factors, 
                "train_images_dir":hparams.train_images_dir,
                "height":hparams.height,
                "width":hparams.width,
                "valid_images_dir":hparams.valid_images_dir,
                "batch_size":self.train_batch_size, 
                "checkpoint_dir":self.checkpoint_dir,
                "save_dir": hparams.save_dir,
                "learning_rate": self.lr,
                "aug_name": self.aug_name,
                "model_name": self.model_name,
                "scheduler_name": self.scheduler_name,
                "optimizer_name": self.optimizer_name,
                "epochs": self.epochs,
                "epoch_length": self.epoch_length,
                "load_model_only":self.load_model_only,
                "use_amp": self.use_amp
            }
        }
