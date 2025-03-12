import numpy as np
import torch
from mmcv.ops.upfirdn2d import downsample2d
from torch import nn
import torch.nn.functional as F
import math
from model.network import setup_filter, EqualizedLinear, MappingNetwork
from model.stylegan2 import EqualizedConv2d


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = torch.sqrt(y + 1e-8)  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.conv_clamp = conv_clamp
        self.fromrgb = nn.Sequential(EqualizedConv2d(img_channels, in_channels, kernel_size=1),
                                     nn.LeakyReLU())
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = nn.Sequential(
            EqualizedConv2d(in_channels + mbstd_num_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU())
        self.fc = nn.Sequential(EqualizedLinear(in_channels * (resolution ** 2), in_channels),
                                nn.LeakyReLU())
        self.out = EqualizedLinear(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap):
        dtype = torch.float32
        x = x.to(dtype=dtype)
        x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = x.clamp(-self.conv_clamp, self.conv_clamp)
        x = self.fc(x.flatten(1))
        x = self.out(x)
        # Conditioning.
        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / math.sqrt(self.cmap_dim))

        return x


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 tmp_channels,  # Number of intermediate channels.
                 out_channels,  # Number of output channels.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 first_layer_idx,  # Index of the first layer.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.num_layers = 0

        self.fromrgb = nn.Sequential(EqualizedConv2d(img_channels, tmp_channels, kernel_size=1),
                                     nn.LeakyReLU())

        self.conv0 = nn.Sequential(EqualizedConv2d(tmp_channels, tmp_channels, kernel_size=3, padding=1),
                                   nn.LeakyReLU())
        self.conv1 = nn.Sequential(EqualizedConv2d(tmp_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, img):
        y = self.fromrgb(img)
        y = y.clamp(-self.conv_clamp, self.conv_clamp)
        x = x + y if x is not None else y
        img = downsample2d(img, self.resample_filter)
        x = self.conv0(x)
        x = F.leaky_relu(x)
        x = x.clamp(-self.conv_clamp, self.conv_clamp)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=0.5, mode='nearest')
        x = F.leaky_relu(x)
        x = x.clamp(-self.conv_clamp, self.conv_clamp)
        return x, img


class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, img_channels=img_channels, conv_clamp=conv_clamp)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4,
                                        img_channels=img_channels, conv_clamp=conv_clamp)

    def forward(self, img, c):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img)
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x
