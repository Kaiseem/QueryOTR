import torch
import torch.nn as nn
from einops import rearrange

class ReconLoss(nn.Module):
    def __init__(self,image_size=192,crop_width=32,loss_type='mse'):
        super(ReconLoss, self).__init__()
        assert loss_type in ['l1','mse']
        mask = torch.ones((image_size, image_size))
        mask[crop_width:-crop_width, crop_width:-crop_width] = 0
        self.mask=mask.view(-1).long().cuda()
        self.outer_index=self.mask==1
        if loss_type=='l1':
            self.loss=nn.L1Loss()
        else:
            self.loss=nn.MSELoss()

    def forward(self,input_fake, input_real):
        input_fake = rearrange(input_fake, 'b c w h -> b (w h) c')[:, self.outer_index]
        input_real = rearrange(input_real, 'b c w h -> b (w h) c')[:, self.outer_index]
        return self.loss(input_fake,input_real)

