import torch
from models.vgg16_extractor import VGG16Extractor
import torch.nn.functional as F

class LossCompute(object):
    def __init__(self, feature_extractor, device="cuda"):
        self.device = device
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

    def gram_matrix(self, feature_matrix):
        (batch, channel, h, w) = feature_matrix.size()
        feature_matrix = feature_matrix.view(batch, channel, h * w)
        feature_matrix_t = feature_matrix.transpose(1, 2)
        gram = torch.bmm(feature_matrix, feature_matrix_t) / (channel * h * w)
        return gram

    def loss_style(self, output, vgg_gt):
        loss = 0.0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss

    def l1(self, y_true, y_pred):
        if len((y_true).shape) == 4:
            return torch.mean(abs(y_pred-y_true),[1,2,3])
        elif len((y_true).shape) == 3:
            return torch.mean(abs(y_pred-y_true),[1,2])
        else:
            raise "ERROR Calculating l1 loss"
            
    def PSNR(self, y_true, y_pred):
        return - 10.0 * torch.log(torch.mean((y_pred - y_true)**2)) / torch.log(torch.tensor(10.0, dtype=torch.float, requires_grad=False)) 
      
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
      
    def loss_tv(self, mask, y_comp):
        kernel = torch.ones((3, 3, mask.shape[1], mask.shape[1]), requires_grad=False).to(self.device)
        dilated_mask = F.conv2d(1-mask, kernel, padding=1)

        dilated_mask = torch.tensor(dilated_mask> 0, dtype=torch.float, requires_grad=False).to(self.device)
        P = dilated_mask * y_comp

        a = self.l1(P[:,:,:,1:], P[:,:,:,:-1])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b
      
    def loss_hole(self, mask, y_true, y_pred):
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        return self.l1(mask * y_true, mask * y_pred)
      
    def loss_total(self, mask):
        def loss(y_true, y_pred):
            y_comp = mask * y_true + (1-mask) * y_pred

            vgg_out = self.feature_extractor(y_pred)
            vgg_gt = self.feature_extractor(y_true)
            vgg_comp = self.feature_extractor(y_comp)

            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_out, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)

            return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6

        return loss