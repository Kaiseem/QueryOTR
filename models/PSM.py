import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

# Equivalent implementation in the paper for efficiency
class PatchSmoothingModule(nn.Module):
    def __init__(self, embed_dim=768, out_chans=3, input_size=128, output_size=192, patch_size=16, overlap_size=8,
                 bias=True):
        super().__init__()
        self.use_bias = bias
        self.patch_size = patch_size
        self.input_size = input_size
        self.output_size = output_size

        self.embed_dim = embed_dim
        patch_size = patch_size
        kernel_size = patch_size + overlap_size * 2
        padding_size = overlap_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, bias=False, kernel_size=kernel_size, stride=patch_size,
                                       padding=padding_size)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(1, out_chans, kernel_size, kernel_size),
                                           requires_grad=True)
            nn.init.constant_(self.bias, 0)

        self.mask = torch.ones(1, 1, output_size // patch_size, output_size // patch_size)
        p = ((output_size - input_size) // 2) // patch_size
        self.mask[:, :, p:-p, p:-p] = 0

        self.mask_weight = F.conv_transpose2d(self.mask.detach(), torch.ones([1, out_chans, kernel_size, kernel_size]),
                                              bias=None, stride=patch_size, padding=padding_size)
        self.mask_weight[self.mask_weight != 0] = 1 / self.mask_weight[self.mask_weight != 0]
        self.patch_size = patch_size
        self.padding_size = padding_size

    def forward(self, x, gt_inner):
        assert x.size(1) == (self.output_size // self.patch_size) ** 2
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.output_size // self.patch_size)
        x = self.proj(x)

        if self.use_bias:
            bias = F.conv_transpose2d(self.mask.detach().to(x.device), self.bias, bias=None, stride=self.patch_size,
                                      padding=self.padding_size)
            x = x + bias
        x = x * self.mask_weight.to(x.device)
        p = (self.output_size - self.input_size) // 2
        x[:, :, p:-p, p:-p] = gt_inner[:, :, p:-p, p:-p]

        return x

# Original implementation in the paper
class PatchSmoothingModule_v2(nn.Module):
    def __init__(self, embed_dim=768, out_chans=3, input_size=128, output_size=192, patch_size=16, overlap_size=8,
                 bias=True):
        super().__init__()
        self.use_bias = bias
        self.patch_size = patch_size
        self.input_size = input_size
        self.output_size = output_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.overlap_size = overlap_size

        patch_size = patch_size
        self.kernel_size = kernel_size = patch_size + overlap_size * 2
        padding_size = overlap_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, bias=bias, kernel_size=kernel_size, stride=kernel_size,
                                       padding=0)

        self.patch_size = patch_size
        self.padding_size = padding_size

    def forward(self, x, gt_inner):
        assert x.size(1) == (self.output_size // self.patch_size) ** 2
        x = rearrange(x, 'b (h w) c -> (b h w) c', h=self.output_size // self.patch_size)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)  # (b h w) c 32 32
        x = rearrange(x, '(b h w) c p1 p2-> b c h w p1 p2', h=self.output_size // self.patch_size,
                      w=self.output_size // self.patch_size)
        output_patches = x

        output = torch.zeros([x.size(0), self.out_chans, self.output_size + 2 * self.overlap_size,
                              self.output_size + 2 * self.overlap_size])

        mask_weight = torch.zeros(
            [1, 1, self.output_size + 2 * self.overlap_size, self.output_size + 2 * self.overlap_size])
        for i in range(self.output_size // self.patch_size):
            for j in range(self.output_size // self.patch_size):
                output[:, :, i * self.patch_size: (i + 1) * self.patch_size + self.overlap_size * 2,
                j * self.patch_size: (j + 1) * self.patch_size + self.overlap_size * 2] += x[:, :, i, j, :, :]
                mask_weight[:, :, i * self.patch_size: (i + 1) * self.patch_size + self.overlap_size * 2,
                j * self.patch_size: (j + 1) * self.patch_size + self.overlap_size * 2] += 1

        mask_weight[mask_weight != 0] = 1 / mask_weight[mask_weight != 0]
        output = output * mask_weight
        output = output[:, :, self.overlap_size:-self.overlap_size, self.overlap_size:-self.overlap_size]
        p = (self.output_size - self.input_size) // 2
        output[:, :, p:-p, p:-p] = gt_inner[:, :, p:-p, p:-p]
        return output, output_patches

if __name__ == '__main__':
    m1 = PatchSmoothingModule()
    m2 = PatchSmoothingModule_v2()
    x1 = torch.randn([1, 144, 768])
    x2 = torch.randn([1, 3, 192, 192])
    y1 = m1(x1, x2)
    y2, _ = m2(x1, x2)
    print(y1.size(), y2.size())

