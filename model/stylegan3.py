import torch.nn.functional as F
import numpy as np
import scipy
import torch
from mmcv.ops import upfirdn2d

from model.network import EqualizedLinear, MappingNetwork


def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None,
                        flip_filter=False):
    if isinstance(padding, int):
        padding = [padding, padding]
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    if b is not None:
        x = x + b.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])
    if clamp is not None and clamp > 0:
        x = x.clamp(-clamp, clamp)
    x = upfirdn2d(x, fu, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)  # Upsample.
    x = F.leaky_relu(x, slope)
    if gain is not None and gain != 1:
        x = x * gain
    if clamp is not None and clamp > 0:
        x = x.clamp(-clamp, clamp)
    x = upfirdn2d(x, fd, down=down, flip_filter=flip_filter)  # Downsample.

    return x


def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels]
        demodulate=True,  # Apply weight demodulation?
        padding=0,  # Padding: int or [padH, padW]
        input_gain=None,
        # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w.to(x.device), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class SynthesisInput(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 channels,  # Number of output channels.
                 size,  # Output spatial size: int or [width, height].
                 sampling_rate,  # Output sampling rate.
                 bandwidth,  # Output bandwidth.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = EqualizedLinear(w_dim, 4)
        self.register_buffer('transform', torch.eye(3, 3))  # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0)  # [batch, row, col]
        freqs = self.freqs.unsqueeze(0)  # [batch, channel, xy]
        phases = self.phases.unsqueeze(0)  # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w)  # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat(
            [w.shape[0], 1, 1])  # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1]  # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat(
            [w.shape[0], 1, 1])  # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2]  # t'_x
        m_t[:, 1, 2] = -t[:, 3]  # t'_y
        transforms = m_r @ m_t @ transforms  # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]],
                                                align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()
        x = x.permute(0, 3, 1, 2)  # [batch, channel, height, width]
        return x


class SynthesisLayer(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 is_torgb,  # Is this the final ToRGB layer?
                 is_critically_sampled,  # Does this layer use critical sampling?
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 in_size,  # Input spatial size: int or [width, height].
                 out_size,  # Output spatial size: int or [width, height].
                 in_sampling_rate,  # Input sampling rate (s).
                 out_sampling_rate,  # Output sampling rate (s).
                 in_cutoff,  # Input cutoff frequency (f_c).
                 out_cutoff,  # Output cutoff frequency (f_c).
                 in_half_width,  # Input transition band half-width (f_h).
                 out_half_width,  # Output Transition band half-width (f_h).

                 # Hyperparameters.
                 conv_kernel=3,  # Convolution kernel size. Ignored for final the ToRGB layer.
                 filter_size=6,  # Low-pass filter size relative to the lower resolution when up/downsampling.
                 lrelu_upsampling=2,  # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
                 use_radial_filters=False,
                 # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
                 conv_clamp=256,  # Clamp the output to [-X, +X], None = disable clamping.
                 magnitude_ema_beta=0.999,  # Decay rate for the moving average of input magnitudes.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.affine = EqualizedLinear(self.w_dim, self.in_channels)
        self.weight = torch.nn.Parameter(
            torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))


        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width * 2, fs=self.tmp_sampling_rate))

        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width * 2, fs=self.tmp_sampling_rate,
            radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor
        pad_total += self.up_taps + self.down_taps - 2
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, update_emas=False):
        if update_emas:
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain
        x = modulated_conv2d(x=x, w=self.weight, s=styles,
                             padding=self.conv_kernel - 1, demodulate=(not self.is_torgb), input_gain=input_gain)
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = _filtered_lrelu_ref(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
                                up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain,
                                slope=slope, clamp=self.conv_clamp)

        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_layers=14,  # Total number of layers, excluding Fourier features and ToRGB.
                 num_critical=2,  # Number of critically sampled layers at the end.
                 first_cutoff=2,  # Cutoff frequency of the first layer (f_{c,0}).
                 first_stopband=2 ** 2.1,  # Minimum stopband of the first layer (f_{t,0}).
                 last_stopband_rel=2 ** 0.3,  # Minimum stopband of the last layer, expressed relative to the cutoff.
                 margin_size=10,  # Number of additional pixels outside the image.
                 output_scale=0.25,  # Scale factor for the output image.
                 ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2  # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel  # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents  # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents  # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution))))  # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs  # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        # Construct layers.
        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled,
                in_channels=int(channels[prev]), out_channels=int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx])
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, update_emas=False):
        ws = ws.to(torch.float32).unbind(dim=1)
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, update_emas)
        if self.output_scale != 1:
            x = x * self.output_scale
        x = x.to(torch.float32)
        return x


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

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        ws = self.mapping(z, c, truncation_psi, truncation_cutoff, update_emas)
        img = self.synthesis(ws, update_emas)
        return img