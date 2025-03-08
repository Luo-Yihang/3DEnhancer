# From https://github.com/Fanghua-Yu/SUPIR/blob/master/SUPIR/modules/SUPIR_v0.py

import torch
import torch as th
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # return super().forward(x.float()).type(x.dtype)
        return super().forward(x)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ZeroSFT(nn.Module):
    def __init__(self, label_nc, norm_nc, nhidden=128, norm=True, mask=False, zero_init=True):
        super().__init__()

        # param_free_norm_type = str(parsed.group(1))
        ks = 3
        pw = ks // 2

        self.norm = norm
        if self.norm:
            self.param_free_norm = normalization(norm_nc)
        else:
            self.param_free_norm = nn.Identity()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        
        if zero_init:
            self.zero_mul = zero_module(nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw))
            self.zero_add = zero_module(nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw))
        else:
            self.zero_mul = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.zero_add = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, c, h, control_scale=1):
        h_raw = h
        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        h = self.param_free_norm(h) * (gamma + 1) + beta

        return h * control_scale + h_raw * (1 - control_scale)