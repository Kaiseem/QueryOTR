import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import itertools
import datetime

torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from datasets import ImageDataset
from util.misc import cosine_scheduler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='scenery')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-4)
parser.add_argument('--warnup_epoch', type=int, default=10)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--half_precision', default=False, type=bool)

parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=192)
parser.add_argument('--enc_ckpt_path', type=str, default='mae_pretrain_vit_base.pth')
parser.add_argument('--dec_depth', type=int, default=4)

parser.add_argument('--data_root', type=str, default='E:/data3/train')
parser.add_argument('--normlize_target', default=True, type=bool, help='normalized the target patch pixels')
parser.add_argument('--patch_mean', type=float, default=0.5044838)
parser.add_argument('--patch_std', type=float, default=0.1355051)

from models.VITGen import TransGen
from models.CNNDis import MsImageDis
from losses import SetCriterion
from engine import train_one_epoch, train_one_epoch_warmup

if __name__ == '__main__':
    opts = parser.parse_args()

    train_dataset = ImageDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size,
                                               num_workers=opts.num_workers, persistent_workers=opts.num_workers > 0,
                                               shuffle=True, pin_memory=True)

    gen = TransGen(opts=opts, enc_ckpt_path=opts.enc_ckpt_path).cuda()
    cnn_dis = MsImageDis().cuda()

    g_param_dicts = [
        {"params": [p for n, p in gen.named_parameters() if 'conv_offset_mask' not in n and not 'transformer_encoder' in n], "lr_scale": 1},
        {"params": [p for n, p in gen.named_parameters() if 'conv_offset_mask' in n], "lr_scale": 0.1},
        {"params": [p for n, p in gen.named_parameters() if 'transformer_encoder' in n], "lr_scale": 1}
    ]

    opt_g = torch.optim.Adam(g_param_dicts, lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.Adam(itertools.chain(cnn_dis.parameters()), lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    lr_schedule_values = cosine_scheduler(opts.lr, opts.min_lr, opts.max_epoch, len(train_loader),
                                          warmup_epochs=opts.warnup_epoch, warmup_steps=-1)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + '_' + opts.name
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    visdir = os.path.join(logdir, "visuals")
    for d in [logdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)
    opts.visdir = visdir
    opts.ckptdir = ckptdir

    if opts.half_precision:
        g_grad_scaler = torch.cuda.amp.GradScaler()
    else:
        g_grad_scaler = None

    criterion = SetCriterion(opts)

    iteration = 1
    for epoch in range(opts.max_epoch):
        # warm up the learning rate
        if lr_schedule_values is not None and epoch < opts.warnup_epoch:
            for i, param_group in enumerate(opt_g.param_groups):
                param_group["lr"] = lr_schedule_values[iteration] * param_group["lr_scale"]
            for i, param_group in enumerate(opt_d.param_groups):
                param_group["lr"] = lr_schedule_values[iteration]
        else:
            for i, param_group in enumerate(opt_g.param_groups):
                param_group["lr"] = opts.lr * param_group["lr_scale"]
            for i, param_group in enumerate(opt_d.param_groups):
                param_group["lr"] = opts.lr

        if epoch < opts.warnup_epoch:
            train_one_epoch_warmup(opts, gen, criterion, train_loader, opt_g, torch.device('cuda'), epoch,
                                   g_grad_scale=g_grad_scaler)
        else:
            train_one_epoch(opts, gen, cnn_dis, criterion, train_loader, opt_g, opt_d, torch.device('cuda'), epoch,
                            g_grad_scale=g_grad_scaler)

        iteration += len(train_loader)

        if (epoch + 1) % 10 == 0 and epoch > 50:
            torch.save({'gen': gen.state_dict()}, os.path.join(ckptdir, f'{epoch + 1}.pth'))
            torch.save({'gen': gen.state_dict()}, os.path.join(ckptdir, f'latest.pth'))
