from utils_helper.image_helpers import unorm_batch
from PIL import Image
import torch

loss_components = ["loss_hole","loss_valid","loss_perceptual","loss_style_out","loss_style_comp","loss_tv"]

def neptune_metrics_completion(trainer, npt_logger):
    def print_metrics(engine):
        metrics = engine.state.metrics
#         print("Validation Results - Epoch: {}  Avg loss: {:.5f} Avg psnr: {:.5f}"
#               .format(trainer.state.epoch, metrics['loss']["loss_sum"], metrics['psnr']))
        npt_logger.experiment.log_metric('valid/avg_loss', trainer.state.epoch, metrics['loss']["avg_loss"]) 
        for loss_comp in loss_components:
            npt_logger.experiment.log_metric(f"valid/{loss_comp}", trainer.state.epoch, metrics['loss'][loss_comp]) 
    
        npt_logger.experiment.log_metric('valid/psnr', trainer.state.epoch, metrics['psnr']) 
    return print_metrics

def score_function(engine):
    return engine.state.metrics['psnr']

def neptune_train_metrics_completion(npt_logger):
    def print_train_metrics_completion(engine):
        metrics = engine.state.metrics
#         print("Train Results - Epoch: {}  Avg loss: {:.5f}"
#               .format(engine.state.epoch, metrics['train_avg_loss']))
        npt_logger.experiment.log_metric('train/avg_loss', engine.state.epoch, metrics["train_avg_loss"]) 
    return print_train_metrics_completion

def neptune_lr_iteration(optimizer,npt_logger):
    def print_lr(engine):
#         print((engine.state.iteration - 1, optimizer.param_groups[0]['lr']))
        npt_logger.experiment.log_metric('lr', engine.state.iteration - 1, optimizer.param_groups[0]['lr']) 
    return print_lr

def neptune_training_iteration(npt_logger):
    def print_training_iteration(engine):
        iteration = engine.state.iteration
#         print(f"Epoch[{engine.state.epoch}] Iteration[{iteration}] Loss: {engine.state.output['loss']}")
        npt_logger.experiment.log_metric('train/loss', iteration, engine.state.output['loss']) 
        for key, value in engine.state.output["dict_losses"].items():
            npt_logger.experiment.log_metric(f"train/{key}", iteration, value)
    return print_training_iteration

def neptune_visualise_training_iteration(npt_logger):
    def visualise_training_iteration(engine):
        mask_imgs = unorm_batch(engine.state.output["mask_image"])[0]
        predicted =unorm_batch(engine.state.output['predicted'])[0]
        target = unorm_batch(engine.state.output['target'])[0]
        resultant_img = torch.cat([mask_imgs, predicted, target],axis=1).detach().cpu().numpy()
#         display(Image.fromarray(resultant_img))
        npt_logger.experiment.log_image('training_sample', resultant_img)
    return visualise_training_iteration

def neptune_visualise_validation_iteration(npt_logger):
    def visualise_validation_iteration(engine):
        mask_imgs = unorm_batch(engine.state.output["mask_image"])[0]
        predicted =unorm_batch(engine.state.output['predicted'])[0]
        target = unorm_batch(engine.state.output['target'])[0]
        resultant_img = torch.cat([mask_imgs, predicted, target],axis=1).detach().cpu().numpy()
#         display(Image.fromarray(resultant_img))
        npt_logger.experiment.log_image('valid_sample', resultant_img)
    return visualise_validation_iteration