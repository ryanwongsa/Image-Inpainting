from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.base_logger import BaseLogger
from ignite.contrib.handlers.neptune_logger import *
from logger.base.utils import *
from logger.neptune_utils import *
import os

class MyNeptuneLogger(BaseLogger):
    
    def __init__(self, project_name, name, params, tags = [], prefix_name="best", n_saved=5, save_dir="../saved_models", log_every=50):
        self.npt_logger = NeptuneLogger(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                           project_name=project_name,
                           name=name,
                           params=params,
                           tags=tags)

        super().__init__(prefix_name=prefix_name, n_saved=n_saved, save_dir=save_dir, log_every=log_every)
    
    def _add_train_events(self, model = None, optimizer=None, scheduler=None, metrics={}, add_pbar=False):
        
        iteration_events = [
            neptune_training_iteration(self.npt_logger),
            neptune_lr_iteration(optimizer, self.npt_logger)
        ]

        completion_events = [
            neptune_train_metrics_completion(self.npt_logger)
        ]
        self._add_train_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events,
                "metrics": metrics,
                "add_pbar":add_pbar
            }
        )
        
        self._add_custom_train_iteration_handler(neptune_visualise_training_iteration(self.npt_logger), 200)
        
#         self.npt_logger.attach(self.trainer,
#             log_handler=GradsScalarHandler(model),
#             event_name=Events.ITERATION_COMPLETED(every=self.log_every))
        
#         self.npt_logger.attach(self.trainer,
#             log_handler=WeightsScalarHandler(model),
#             event_name=Events.ITERATION_COMPLETED(every=self.log_every))
#         self.npt_logger.attach(self.trainer,
#             log_handler=OptimizerParamsHandler(optimizer),
#             event_name=Events.ITERATION_COMPLETED(every=self.log_every))
        
    
    
    def _add_eval_events(self, validloader=None, metrics = {}, to_save= {}, add_pbar=False):
        iteration_events = []
        
        completion_events = [
            neptune_metrics_completion(self.trainer, self.npt_logger)
        ]
        
        self._add_evaluation_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events,
                "metrics": metrics,
                "validloader":validloader,
                "to_save": to_save,
                "score_function": score_function,
                "add_pbar":add_pbar
            }
        )
        
        self._add_custom_eval_iteration_handler(neptune_visualise_validation_iteration(self.npt_logger), 200)
        
        handler = Checkpoint(to_save, NeptuneSaver(self.npt_logger), n_saved=self.n_saved,
                     filename_prefix=self.prefix_name, score_function=score_function,
                     score_name="score",
                     global_step_transform=global_step_from_engine(self.trainer))
        self.evaluator.add_event_handler(Events.COMPLETED, handler)
        