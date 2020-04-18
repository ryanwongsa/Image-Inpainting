from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

def add_iteration_handlers(engine, method_events = [], log_every=5):
    for method_event in method_events:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=log_every), method_event)

def add_progress_bar(engine):
    pbar = ProgressBar(persist=True)
    pbar.attach(engine)

def add_completion_handlers(engine, method_events = []):
    for method_event in method_events:
        engine.add_event_handler(Events.EPOCH_COMPLETED, method_event)
        
def save_checkpoint(trainer, evaluator, to_save, score_function, save_dir, n_saved, prefix_name):
    if save_dir is not None and score_function is not None:
        handler = Checkpoint(to_save, DiskSaver(save_dir, require_empty=False), n_saved=n_saved,
                             filename_prefix=prefix_name, score_function=score_function, score_name="score", 
                             global_step_transform=global_step_from_engine(trainer))

        evaluator.add_event_handler(Events.COMPLETED, handler)
        
def evaluate_on_metrics(evaluator, dataloader):
    def evaluate_metrics(engine):
        evaluator.run(dataloader)
    return evaluate_metrics

def add_evaluation_metrics(trainer, evaluator, dataloader, metrics):
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate_on_metrics(evaluator, dataloader))
    
def add_training_metrics(trainer, metrics):
    for name, metric in metrics.items():
        metric.attach(trainer, name)