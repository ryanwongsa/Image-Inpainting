import torch
import torch.nn.functional as F
from torch import nn
from models.pconv_decoder import PConvDecoder
from models.pconv_encoder import PConvEncoder

class PConvUNet(nn.Module):
    def __init__(self, channels=3):
        super(PConvUNet, self).__init__()
        self.encoder1 = PConvEncoder(channels, 64, 7, bn=False)
        self.encoder2 = PConvEncoder(64, 128, 5)
        self.encoder3 = PConvEncoder(128, 256, 5)
        self.encoder4 = PConvEncoder(256, 512, 3)
        self.encoder5 = PConvEncoder(512, 512, 3)
        self.encoder6 = PConvEncoder(512, 512, 3)
        self.encoder7 = PConvEncoder(512, 512, 3)
        self.encoder8 = PConvEncoder(512, 512, 3)
        
        self.decoder1 = PConvDecoder(512+512, 512, 3)
        self.decoder2 = PConvDecoder(512+512, 512, 3)
        self.decoder3 = PConvDecoder(512+512, 512, 3)
        self.decoder4 = PConvDecoder(512+512, 512, 3)
        self.decoder5 = PConvDecoder(512+256, 256, 3)
        self.decoder6 = PConvDecoder(256+128, 128, 3)
        self.decoder7 = PConvDecoder(128+64, 64, 3)
        self.decoder8 = PConvDecoder(64+3, 3, 3, bn=False, act=False, return_mask=False)
        self.convfinal = nn.Conv2d(3, 3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, inputs_img, inputs_mask):
        e_conv1, e_mask1 = self.encoder1(inputs_img, inputs_mask)
        e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1)
        e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2)
        e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3)
        e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4)
        e_conv6, e_mask6 = self.encoder6(e_conv5, e_mask5)
        e_conv7, e_mask7 = self.encoder7(e_conv6, e_mask6)
        e_conv8, e_mask8 = self.encoder8(e_conv7, e_mask7)
        
        
        d_conv9, d_mask9   = self.decoder1(e_conv8, e_mask8, e_conv7, e_mask7)
        d_conv10, d_mask10 = self.decoder2(d_conv9, d_mask9, e_conv6, e_mask6)
        d_conv11, d_mask11 = self.decoder3(d_conv10, d_mask10, e_conv5, e_mask5)
        d_conv12, d_mask12 = self.decoder4(d_conv11, d_mask11, e_conv4, e_mask4)
        d_conv13, d_mask13 = self.decoder5(d_conv12, d_mask12, e_conv3, e_mask3)
        d_conv14, d_mask14 = self.decoder6(d_conv13, d_mask13, e_conv2, e_mask2)
        d_conv15, d_mask15 = self.decoder7(d_conv14, d_mask14, e_conv1, e_mask1)
        output = self.decoder8(d_conv15, d_mask15, inputs_img, inputs_mask)
#         output = self.convfinal(output)
#         output = self.sigmoid(output)
        return output