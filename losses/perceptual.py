from torchvision.models.vgg import vgg19

import torch
import torch.nn as nn


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class PerceptualLoss(nn.Module):
    '''
    same as https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py
    '''
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    #     self.vgg_model= vgg16(pretrained=True)
    #     self.vgg_model.features.__delitem__(23)#delete maxpool
    #     self.vgg_model.features.__delitem__(-1)#delete maxpool
    #     self.instance_norm=nn.InstanceNorm2d(512,affine=False)
    #     self.vgg_model.eval()
    #     for param in self.vgg_model.parameters():
    #         param.requires_grad = False
    #     self.cuda()
    #
    # def forward(self, input_fake, input_real):
    #     return torch.mean((self.instance_norm(self.vgg_model.features(input_fake)) - self.instance_norm(self.vgg_model.features(input_real))) ** 2)


