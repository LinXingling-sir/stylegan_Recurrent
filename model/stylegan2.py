import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.network import EqualizedWeight, EqualizedLinear, setup_filter, MappingNetwork


class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):
        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=True)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class SynthesisLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this layer.
                 kernel_size=3,  # Convolution kernel size.
                 up=1,  # Integer upsampling factor.
                 use_noise=True,  # Enable noise input?
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.

                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.affine = EqualizedLinear(w_dim, in_channels)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.noise_strength = torch.nn.Parameter(torch.tensor(0.1))  # 初始值0.1
        self.upsample = torch.nn.Upsample(scale_factor=up, mode='nearest')
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size, demodulate=True)

    def forward(self, x, w, gain=1):
        styles = self.affine(w)
        noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        x = self.upsample(x)

        x = x.add_(noise)
        x = self.conv(x, styles)
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = x + self.bias.to(x.dtype).reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
        x = F.leaky_relu(x, 0.2)
        x = x * float(math.sqrt(2)) * gain
        if act_clamp > 0:
            x = x.clamp(-act_clamp, act_clamp)
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color channels.
                 is_last,  # Is this the last block?
                 resample_filter=[1, 3, 3, 1],
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.num_conv = 0
        self.num_torgb = 0
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        conv_clamp=conv_clamp)
            self.num_conv += 1
        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp)
        self.num_conv += 1
        self.torgb = ToRGB(w_dim, out_channels)
        self.num_torgb += 1

    def forward(self, x, img, ws):
        w_iter = iter(ws.unbind(dim=1))
        if self.in_channels == 0:
            x = self.const
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter))
        else:
            x = self.conv0(x, next(w_iter))
            x = self.conv1(x, next(w_iter))
        from mmcv.ops import upsample2d
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        y = self.torgb(x, next(w_iter))
        y = y.to(dtype=torch.float32)
        img = img.add_(y) if img is not None else y
        return x, img


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(math.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                   img_channels=img_channels, is_last=is_last)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws):
        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv
        x = img = None

        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws)
        return img


class Generator(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws)
        self.ws = None

    def getws(self):
        return self.ws if self.ws is not None else None

    def forward(self, z, c):
        ws = self.mapping(z, c)
        self.ws = ws
        img = self.synthesis(ws)
        return img