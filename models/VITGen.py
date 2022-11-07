import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import get_sinusoid_encoding_table, CorssAttnBlock
from .VIT import *
from .QEM import QueryExpansionModule
from .PSM import PatchSmoothingModule

class TransGen(nn.Module):
    def __init__(self,opts, enc_ckpt_path=None):
        super(TransGen, self).__init__()
        self.output_size=opts.output_size
        self.input_size=opts.input_size
        self.patch_size=16 # same as ViT-B
        hidden_num=768 # same as ViT-B

        # initialize the weight of decoder, psm and qem
        self.qem=QueryExpansionModule(hidden_num=hidden_num,input_size=self.input_size,outout_size=self.output_size,patch_size=self.patch_size)
        self.transformer_decoder=nn.ModuleList([
            CorssAttnBlock(
                dim=hidden_num, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                init_values=0., window_size= None)
            for _ in range(opts.dec_depth)])
        self.psm=PatchSmoothingModule(patch_size=16,out_chans=3,embed_dim=hidden_num)
        self.apply(self._init_weights)

        # initialize the weight of encoder using pretrain checkpoint

        self.transformer_encoder = vit_base_patch16(pretrained=True, img_size=224, init_ckpt=enc_ckpt_path)
        #vit_base_patch16(pretrain=True,  init_ckpt=enc_ckpt_path, img_size=self.input_size)

        self.enc_image_size=224

        # initialize the weight of encoder using pretrain checkpoint
        self.pos_embed = get_sinusoid_encoding_table(12**2, hidden_num)
        self.inner_index, self.outer_index=self.get_index()

    def get_index(self):
        input_query_width=self.input_size//self.patch_size
        output_query_width=self.output_size//self.patch_size
        mask=torch.ones(size=[output_query_width,output_query_width]).long()
        pad_width=(output_query_width-input_query_width)//2
        mask[pad_width:-pad_width,pad_width:-pad_width] = 0
        mask=mask.view(-1)
        return mask==0,mask==1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, samples):
        if type(samples) is not dict:
            samples={'input':samples, 'gt_inner':F.pad(samples,(32,32,32,32))}
        x = samples['input']
        
        gt_inner = samples['gt_inner']

        b,c,w,h=x.size()

        assert w==128 and h==128
        padded_x = F.pad(x, (48, 48, 48, 48), mode='reflect')
        vit_mask = torch.ones(size=(14, 14)).long()
        vit_mask[3:-3, 3:-3] = 0

        vit_mask = vit_mask.view(-1).expand(b, -1).contiguous().bool()

        src = self.transformer_encoder.forward_features(padded_x, vit_mask)  # b n c

        query_embed=self.qem(src)

        full_pos=self.pos_embed.type_as(x).to(x.device).clone().detach().expand(x.size(0),-1,-1)

        tgt_outer=query_embed[:,self.outer_index,:]+full_pos[:,self.outer_index,:]

        for i,dec in enumerate(self.transformer_decoder):
            tgt_outer = dec(tgt_outer, src)

        tgt = torch.zeros_like(query_embed,dtype=torch.float32)

        tgt[:, self.outer_index] = tgt_outer

        fake=self.psm(tgt,gt_inner)
        return fake










