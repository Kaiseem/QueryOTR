
import math
import os
import sys
from typing import Iterable
import torch

import functools
print = functools.partial(print, flush=True)
import util.misc as utils

from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import time

def denorm_img(tensor, opts):
    tensor = rearrange(tensor[0:4], 'b c w h -> b w h c').detach().cpu()
    tensor = tensor * torch.tensor((opts.patch_std,opts.patch_std,opts.patch_std)) + torch.tensor((opts.patch_mean,opts.patch_mean,opts.patch_mean))
    tensor = np.clip(tensor.flatten(0, 1).numpy(), 0, 1)
    return tensor

def train_one_epoch(opts, GEN: torch.nn.Module, DIS: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, gen_opt: torch.optim.Optimizer, dis_opt: torch.optim.Optimizer,
                    device: torch.device, epoch: int, g_grad_scale=None):
    GEN.train()
    DIS.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    display_freq=75

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k,v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k]=v.to(device)
        if g_grad_scale is None:
            gt=samples['ground_truth']
            fake = GEN(samples)
            if i%display_freq==0:
                fig_name = f"{epoch}_{time.time():04f}"
                fig = np.concatenate([denorm_img(gt, opts), denorm_img(fake, opts)], axis=1)
                plt.imsave(os.path.join(opts.visdir, fig_name + '.png'), fig, vmin=0, vmax=1)

            D_loss_dict = criterion.get_dis_loss(fake, gt, DIS)
            D_losses = sum(D_loss_dict[k] * criterion.dis_weight_dict[k] for k in D_loss_dict.keys())

            dis_opt.zero_grad()
            D_losses.backward()
            dis_opt.step()

            G_loss_dict = criterion.get_gen_loss(fake, gt, DIS)
            G_losses = sum(G_loss_dict[k] * criterion.gen_weight_dict[k] for k in G_loss_dict.keys())

            gen_opt.zero_grad()
            G_losses.backward()
            gen_opt.step()

        else:
            with torch.cuda.amp.autocast():
                gt = samples['ground_truth']
                fake = GEN(samples)

            if i % display_freq == 0:
                fig_name = f"{epoch}_{time.time():04f}"
                fig = np.concatenate([denorm_img(gt, opts), denorm_img(fake, opts)], axis=1)
                plt.imsave(os.path.join(opts.visdir, fig_name + '.png'), fig, vmin=0, vmax=1)

            D_loss_dict = criterion.get_dis_loss(fake, gt)
            D_losses = sum(D_loss_dict[k] * criterion.dis_weight_dict[k] for k in D_loss_dict.keys())

            dis_opt.zero_grad()
            D_losses.backward()
            dis_opt.step()

            with torch.cuda.amp.autocast():
                G_loss_dict = criterion.get_gen_loss(fake, gt)
                G_losses = sum(G_loss_dict[k] * criterion.gen_weight_dict[k] for k in G_loss_dict.keys())

            gen_opt.zero_grad()
            g_grad_scale.scale(G_losses).backward()
            g_grad_scale.step(gen_opt)
            g_grad_scale.update()

        metric_logger.update(**G_loss_dict,**D_loss_dict)
        metric_logger.update(lr=dis_opt.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_warmup(opts, GEN: torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, gen_opt: torch.optim.Optimizer,
                    device: torch.device, epoch: int, g_grad_scale=None):


    GEN.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = '(Warning Up!!)Epoch: [{}]'.format(epoch)
    print_freq = 10
    display_freq = 75

    for i,samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        if g_grad_scale is None:
            gt = samples['ground_truth']
            outputs = GEN(samples)
            if i%display_freq==0:
                fig_name = f"{epoch}_{time.time():04f}"
                fig = np.concatenate([denorm_img(gt, opts), denorm_img(outputs, opts)], axis=1)
                plt.imsave(os.path.join(opts.visdir, fig_name + '.png'), fig, vmin=0, vmax=1)
            G_loss_dict = criterion.get_gen_loss(outputs, gt,  warmup=True)
            G_losses = sum(G_loss_dict[k] * criterion.gen_weight_dict[k] for k in G_loss_dict.keys())

            gen_opt.zero_grad()
            G_losses.backward()
            gen_opt.step()
        else:
            gen_opt.zero_grad()
            with torch.cuda.amp.autocast():
                gt = samples['ground_truth']
                outputs = GEN(samples)
                if i % display_freq == 0:
                    fig_name = f"{epoch}_{time.time():04f}"
                    fig = np.concatenate([denorm_img(gt, opts), denorm_img(outputs, opts)], axis=1)
                    plt.imsave(os.path.join(opts.visdir, fig_name + '.png'), fig, vmin=0, vmax=1)
                G_loss_dict = criterion.get_gen_loss(outputs, gt, warmup=True)
                G_losses = sum(G_loss_dict[k] * criterion.gen_weight_dict[k] for k in G_loss_dict.keys())
            g_grad_scale.scale(G_losses).backward()
            g_grad_scale.step(gen_opt)
            g_grad_scale.update()

        metric_logger.update(**G_loss_dict)
        metric_logger.update(lr=gen_opt.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}