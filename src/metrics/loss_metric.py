from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import torch.nn as nn
from math import log10
from collections import defaultdict
class Loss_Metric(Metric):

    def __init__(self, loss_compute, output_transform=lambda x: x, device=None):
        self._loss_values = None
        self._num_examples = None
        self._loss_dict = None
        self.loss_compute = loss_compute
        super(Loss_Metric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._loss_values = 0
        self._num_examples = 0
        self._loss_dict = defaultdict(lambda :0)
        super(Loss_Metric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        masks, y_pred, y_true = output
        ls_fn = self.loss_compute.loss_total(masks)
        loss, dict_loss = ls_fn(y_true, y_pred)
        
        for key, value in dict_loss.items():
            self._loss_dict[key]+=value
        
        self._loss_values += loss.item()
        self._num_examples += y_true.shape[0]

    @sync_all_reduce("_num_examples", "_loss_values", "_loss_dict")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss_Metric must have at least one example before it can be computed.')
        dict_loss = {}
        for key, values in self._loss_dict.items():
            dict_loss[key] = values/self._num_examples
        return {"avg_loss":self._loss_values / self._num_examples, **dict_loss}