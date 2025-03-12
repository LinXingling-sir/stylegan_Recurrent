from math import sqrt
import numpy as np
import torch
from torch import nn

def setup_filter(f, device=torch.device('cpu'), normalize=True, gain=1):
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    if f.ndim == 0:
        f = f[np.newaxis]
    separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    if normalize:
        f /= f.sum()
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class EqualizedWeight(nn.Module):

    def __init__(self, shape):
        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class MappingNetwork(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 num_ws,  # Number of intermediate latents to output, None = do not broadcast.
                 num_layers=8,
                 w_avg_beta=0.998
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        embed_features = w_dim
        self.w_avg_beta = w_avg_beta
        if c_dim == 0:
            embed_features = 0
        layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = nn.Sequential(EqualizedLinear(c_dim, embed_features),
                                       nn.Linear(embed_features, embed_features))
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = nn.Sequential(EqualizedLinear(in_features, out_features),
                                  nn.LeakyReLU())
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        with torch.cuda.amp.autocast(enabled=False):
            if update_emas and self.w_avg_beta is not None:
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg.to(torch.float16), self.w_avg_beta))
        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = (self.w_avg.lerp(x.to(torch.float32), truncation_psi))

            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x
