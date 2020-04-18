from argparse import ArgumentParser, Namespace
from engine import InpaintingEngine
from config_params.parameters import Parameters


def main(hyperparams):
    parameters = Parameters(hyperparams)    
    engine = InpaintingEngine(parameters)
    engine.train({"max_epochs":parameters.epochs, "epoch_length":parameters.epoch_length})
    
if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    
    parser.add_argument('--train_images_dir', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--train_invert_mask', dest='train_invert_mask', action='store_true')

    parser.add_argument('--valid_images_dir', type=str)
    parser.add_argument('--valid_mask_dir', type=str)
    parser.add_argument('--valid_invert_mask', dest='valid_invert_mask', action='store_true')    

    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--height', default=512, type=int)
    parser.add_argument('--width', default=512, type=int)

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--train_bs', default=6, type=int)
    parser.add_argument('--val_bs', default=6, type=int)
    parser.add_argument('--lr', type=float, default=0.0003) 
    
    parser.add_argument('--name', type=str)
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--n_saved', default=5, type=int)
    parser.add_argument('--log_every', default=1, type=int)
    parser.add_argument('--tags', default="", type=str)
    
    parser.add_argument('--loss_hole', type=float, default=6.0) 
    parser.add_argument('--loss_valid', type=float, default=1.0) 
    parser.add_argument('--loss_perceptual', type=float, default=0.05) 
    parser.add_argument('--loss_style_out', type=float, default=120.0) 
    parser.add_argument('--loss_style_comp', type=float, default=120.0) 
    parser.add_argument('--loss_tv', type=float, default=0.1) 
    
    parser.add_argument('--aug_name', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--scheduler_name', default=None, type=str)
    parser.add_argument('--optimizer_name', default=None, type=str)
    
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--epoch_length', default=None, type=int)
    
    parser.add_argument('--add_pbar', dest='add_pbar', action='store_true')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--load_model_only', dest='load_model_only', action='store_true')
    
    
    hyperparams = parser.parse_args()
    
    main(hyperparams)