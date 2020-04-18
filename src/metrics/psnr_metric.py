from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import torch.nn as nn
from math import log10

class PSNR_Metric(Metric):

    def __init__(self, output_transform=lambda x: x, device=None):
        self._psnr_values = None
        self._num_examples = None
        
        self.criterion = nn.MSELoss()

        super(PSNR_Metric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._psnr_values = 0
        self._num_examples = 0
        super(PSNR_Metric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y_true = output
        mse = self.criterion(y_pred, y_true)
        psnr = 10 * log10(1 / mse.item())
        
        self._psnr_values += psnr
        self._num_examples += y_true.shape[0]

    @sync_all_reduce("_num_examples", "_psnr_values")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('PSNR_Metric must have at least one example before it can be computed.')
        return self._psnr_values / self._num_examples