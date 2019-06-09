from torchvision import models
import torch
import torch.nn.functional as F
from torch import nn

class VGG16Extractor(nn.Module):
	def __init__(self):
		super().__init__()
		vgg16 = models.vgg16(pretrained=True)
		self.max_pooling1 = vgg16.features[:5]
		self.max_pooling2 = vgg16.features[5:10]
		self.max_pooling3 = vgg16.features[10:17]

		for i in range(1, 4):
			for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
				param.requires_grad = False

	def forward(self, image):
		results = [image]
		for i in range(1, 4):
			func = getattr(self, 'max_pooling{:d}'.format(i))
			results.append(func(results[-1]))
		return results[1:]