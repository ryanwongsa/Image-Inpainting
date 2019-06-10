import torch
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

class Preprocessor(object):
	def __init__(self, device):
		self.device = device
		self.std = torch.Tensor(STDDEV).type(torch.float).to(self.device)
		self.mean = torch.Tensor(MEAN).type(torch.float).to(self.device)

	def unnormalize(self, x):
	    x = x.transpose(1, 3)
	    x = x * self.std + self.mean
	    return x

	def normalize(self, x):
	    x = (x-self.mean)/self.std
	    x = x.transpose(1, 3)
	    return x