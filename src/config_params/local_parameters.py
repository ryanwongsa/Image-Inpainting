class LocalParameters(object):
    def __init__(self, hparams=None):
        
        self.aug_name = None
        self.model_name = None
        self.scheduler_name = None
        self.optimizer_name = None
        
        self.epochs = 2
        self.epoch_length = 100
        self.train_mask_params = {
            "filepath":  "D:/Repos/image-inpainting/dataset/irregular_mask/irregular_mask/disocclusion_img_mask",
            "height": 512, 
            "width": 512, 
            "invert_mask": False, 
            "channels": 3, 
            "random_seed": None
        }
        self.train_ds_params = {
            "root_dir":  "D:/Repos/image-inpainting/dataset/train_0",
            "height": 512, 
            "width": 512
        }

        self.valid_mask_params = {
            "filepath":  "D:/Repos/image-inpainting/dataset/irregular_mask/irregular_mask/disocclusion_img_mask",
            "height": 512, 
            "width": 512, 
            "invert_mask": False, 
            "channels": 3, 
            "random_seed": None
        }

        self.valid_ds_params = {
            "root_dir":  "D:/Repos/image-inpainting/dataset/train_0",
            "height": 512, 
            "width": 512, 
            "isValidation": True,
        }

        self.num_workers = 0
        self.train_batch_size = 2
        self.val_batch_size = 2
        self.checkpoint_dir = None
        self.save_dir = None
        self.lr = 0.0003

        self.loss_factors = {
            "loss_hole": 6.0, 
            "loss_valid": 1.0,
            "loss_perceptual":0.05,
            "loss_style_out": 120.0,
            "loss_style_comp": 120.0,
            "loss_tv": 0.1
        }
        
        self.add_pbar = True

        self.logger_params = {
            "save_dir" : "../saved_models",
            "n_saved": 5,
            "prefix_name": "best",
            "log_every": 1,
            "project_name": "gatletag/image-inpainting",
            "name": "ignite-image-inpainting",
            "tags": ["test"],
            "params": {
                **self.loss_factors, 
                "train_images_dir": None,
                "height": 512,
                "width": 512,
                "valid_images_dir":None,
                "batch_size":self.train_bs, 
                "checkpoint_dir":self.checkpoint_dir,
                "save_dir": None,
                "learning_rate": self.lr,
                "aug_name": self.aug_name,
                "model_name": self.model_name,
                "scheduler_name": self.scheduler_name,
                "optimizer_name": self.optimizer_name,
                "epochs": self.epochs,
                "epoch_length": self.epoch_length
            }
        }

