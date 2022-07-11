import os
from einops import rearrange
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from models.VITGen import TransGen
import numpy as np
from datasets import ImageDataset
import argparse
import torch_fidelity

parser = argparse.ArgumentParser()
parser.add_argument('--eval', default=True, type=bool)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("-r", "--resume", type=str)
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=192)
parser.add_argument('--dec_depth', type=int, default=4)
parser.add_argument('--normlize_target', default=True, type=bool, help='normalized the target patch pixels')
parser.add_argument('--patch_mean', type=float, default=0.5044838)
parser.add_argument('--patch_std', type=float, default=0.1355051)
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--epoch', type=str, default=None)
opts = parser.parse_args()



def denorm_img(tensor):
    _mean = torch.tensor([opts.patch_mean, opts.patch_mean, opts.patch_mean]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    _std = torch.tensor([opts.patch_std, opts.patch_std, opts.patch_std]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    tensor = tensor * _std.cuda().expand_as(tensor) + _mean.cuda().expand_as(tensor)
    tensor = rearrange(tensor[0:1], 'b c w h -> b w h c').detach().cpu()
    tensor = np.clip(tensor[0].numpy(), 0, 1)
    return tensor

if __name__=='__main__':
    gen = TransGen(opts=opts).cuda()
    logdir = opts.resume
    ckptdir = os.path.join(logdir, "checkpoints")
    if opts.epoch is not None:
        ckpt_name = os.path.join(ckptdir, f'{opts.epoch}.pth')
    else:
        ckpt_name = os.path.join(ckptdir, 'latest.pth')
    assert os.path.isfile(ckpt_name), f'check if existing checkpoint files {ckpt_name}'
    gtdir = os.path.join(logdir, "gt")
    generatedir = os.path.join(logdir, "generated")
    for d in [gtdir, generatedir]:
        os.makedirs(d, exist_ok=True)
    test_dataset = ImageDataset(opts)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
    state_path = os.path.join(ckpt_name)
    state_dict = torch.load(state_path)
    gen.load_state_dict(state_dict['gen'])
    gen.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            for k in test_data.keys():
                if isinstance(test_data[k], torch.Tensor):
                    test_data[k] = test_data[k].cuda()
            name = test_data['name']
            fake = gen(test_data)
            gt = test_data['ground_truth']
            for i in range(fake.size(0)):
                plt.imsave(os.path.join(gtdir, f'{name[i]}.png'), denorm_img(gt[i:i + 1]), vmin=0, vmax=1)
                plt.imsave(os.path.join(generatedir, f'{name[i]}.png'), denorm_img(fake[i:i + 1]), vmin=0, vmax=1)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generatedir,
        input2=gtdir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        verbose=True,
    )