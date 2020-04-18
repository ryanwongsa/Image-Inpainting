from utils_helper.image_helpers import unorm_batch
from PIL import Image

def print_metrics_completion(trainer):
    def print_metrics(engine):
        metrics = engine.state.metrics
        print("Validation Results - Epoch: {}  Avg loss: {:.5f} Avg psnr: {:.5f}"
              .format(trainer.state.epoch, metrics['loss']["avg_loss"], metrics['psnr']))
    return print_metrics

def score_function(engine):
    return engine.state.metrics['psnr']

def print_train_metrics_completion(engine):
    metrics = engine.state.metrics
    print("Train Results - Epoch: {}  Avg loss: {:.5f}"
          .format(engine.state.epoch, metrics['train_avg_loss']))
    
def print_lr_iteration(optimizer):
    def print_lr(engine):
        print((engine.state.iteration - 1, optimizer.param_groups[0]['lr']))
    return print_lr

def print_training_iteration(engine):
    iteration = engine.state.iteration
    print(f"Epoch[{engine.state.epoch}] Iteration[{iteration}] Loss: {engine.state.output['loss']}")
    
def visualise_training_iteration(engine):
    mask_imgs = unorm_batch(engine.state.output["mask_image"])[0]
    predicted =unorm_batch(engine.state.output['predicted'])[0]
    target = unorm_batch(engine.state.output['target'])[0]
    resultant_img = torch.cat([mask_imgs, predicted, target],axis=1).detach.cpu().numpy()
    display(Image.fromarray(resultant_img))
    experiment.log_image('training_sample', resultant_img)