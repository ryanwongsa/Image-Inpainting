from lightning.image_inpainting_system import ImageInpaintingSystem
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import os

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    if hparams.logs_dir is None:
        model = ImageInpaintingSystem(hparams)
    else:
        model = ImageInpaintingSystem.load_from_metrics(
            weights_path=hparams.logs_dir+"/checkpoints/"+ hparams.checkpoint_name,
            tags_csv=hparams.logs_dir+'/meta_tags.csv'
        )

    num_gpus = 1
    
    trainer = Trainer(
        gpus=1,
        train_percent_check=hparams.train_percent_check, 
        val_check_interval=hparams.val_check_interval,
        use_amp=hparams.use_16bit,
        default_save_path=hparams.save_path
    )

    trainer.fit(model)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    
    parent_parser.add_argument(
        '--train_percent_check',
        type=float,
        default=1.0,
    )
    
    parent_parser.add_argument(
        '--val_check_interval',
        type=float,
        default=0.05,
    )
    
    parent_parser.add_argument(
        '--logs_dir',
        type=str
    )
    
    parent_parser.add_argument(
        '--checkpoint_name',
        type=str
    )
    
    parent_parser.add_argument(
        '--save_path',
        type=str
    )
    
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parser = ImageInpaintingSystem.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)