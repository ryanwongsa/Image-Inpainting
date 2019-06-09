import torch
import torch.nn.functional as F
from torch import nn
from models.partial_conv2d import PartialConv2d

class PConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True, act=True, return_mask=True):
        super(PConvDecoder, self).__init__()
        self.upsample_img = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_mask = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,multi_channel=True, return_mask=return_mask)
        self.bn = bn
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.act = act
        self.return_mask = return_mask
        
    def forward(self, img, mask_in, e_conv, e_mask):
        up_img = self.upsample_img(img)
        up_mask = self.upsample_mask(mask_in)
        
        concat_img = torch.cat([e_conv, up_img], dim=1)
        concat_mask = torch.cat([e_mask, up_mask], dim=1)
        if self.return_mask:
            conv, mask = self.pconv(concat_img, concat_mask)
        else:
            conv = self.pconv(concat_img, concat_mask)
        if self.bn:
            conv = self.batchnorm(conv)
        if self.act:
            conv = self.activation(conv)
            
        if self.return_mask:
            return conv, mask
        else:
            return conv