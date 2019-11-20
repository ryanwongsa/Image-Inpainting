from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
import math

class LightningDaliDataloader(PyTorchIterator):
    def __init__(self, pipe, size, batch_size, output_map=["data", "label"], last_batch_padded=False, fill_last_batch=True):
        super().__init__(pipe, size=size, output_map=output_map, last_batch_padded=last_batch_padded, fill_last_batch=fill_last_batch)
        self.dataset_size, self.batch_size = size, batch_size
        self.last_batch_padded = last_batch_padded

    def __len__(self):
        if self.last_batch_padded:
          return self.dataset_size//self.batch_size
        else:
          return math.ceil(self.dataset_size/self.batch_size)