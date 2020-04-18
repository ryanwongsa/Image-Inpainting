from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.base_logger import BaseLogger
from logger.base.utils import *
from logger.print_utils import *

class PrintLogger(BaseLogger):
    
    def __init__(self, project_name, name, params, tags=[],prefix_name="best", n_saved=5, save_dir=None, log_every=5):
        super().__init__(prefix_name=prefix_name, n_saved=n_saved, save_dir=save_dir, log_every=log_every)
            
    def _add_train_events(self, model = None, optimizer=None, scheduler=None, metrics={}):
        
        iteration_events = [
            print_training_iteration,
            print_lr_iteration(optimizer)
        ]

        completion_events = [
            print_train_metrics_completion
        ]
        self._add_train_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events,
                "metrics": metrics
            }
        )
    
    def _add_eval_events(self, validloader=None, metrics = {}, to_save= {}):
        iteration_events = []
        
        completion_events = [
            print_metrics_completion(self.trainer)
        ]
        
        self._add_evaluation_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events,
                "metrics": metrics,
                "validloader":validloader,
                "to_save": to_save,
                "score_function": score_function
            }
        )
