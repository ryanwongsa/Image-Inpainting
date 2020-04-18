from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.utils import *

class BaseLogger(object):
    
    def __init__(self, prefix_name="best", n_saved=5, save_dir=None, log_every=50):
        self.log_every = log_every
        self.save_dir = save_dir
        self.prefix_name = prefix_name
        self.n_saved = n_saved
        self.trainer = None
        self.evaluator = None
    
    def _init_logger(self, trainer, evaluator):
        self.trainer = trainer
        self.evaluator = evaluator
        
    def _add_train_handlers(self, iteration_events = [], completion_events = [], metrics={}, add_pbar=False):
        if add_pbar:
            add_progress_bar(self.trainer)
        add_iteration_handlers(self.trainer, iteration_events, self.log_every)
        add_training_metrics(self.trainer, metrics)
        add_completion_handlers(self.trainer, completion_events)
    
    def _add_evaluation_handlers(self, iteration_events = [], completion_events = [], metrics={}, validloader=None, to_save={}, score_function=None, add_pbar=False):
        
        add_iteration_handlers(self.evaluator, iteration_events, self.log_every)
        if add_pbar:
            add_progress_bar(self.evaluator)
        add_evaluation_metrics(self.trainer, self.evaluator, validloader, metrics)
        save_checkpoint(self.trainer, self.evaluator, to_save, score_function,
                             self.save_dir, self.n_saved, self.prefix_name)
        add_completion_handlers(self.evaluator, completion_events)
        
    def _add_train_events(self, model = None, optimizer=None, scheduler=None, metrics={}):
        raise NotImplementedError
        
    def _add_eval_events(self, validloader=None, metrics = {}, to_save= {}):
        raise NotImplementedError
        
    def _add_custom_train_iteration_handler(self, iteration_event, log_every):
        add_iteration_handlers(self.trainer, [iteration_event], log_every)
        
    def _add_custom_eval_iteration_handler(self, iteration_event, log_every):
        add_iteration_handlers(self.evaluator, [iteration_event], log_every)