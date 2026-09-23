"""FSU-FNO: a frequency-supervised U-FNO for steady-state chip thermal fields.

Architecture: a six-block U-FNO backbone (the last three blocks carry a U-Net
branch) built on axis-factorized spectral convolutions, with the U-Net branch
applied *after* the Fourier and pointwise paths are summed.

One block:
    h        = K_l(x_l) + W_l x_l
    h       <- h + U_l(h)          (only l in {3, 4, 5})
    x_{l+1}  = ReLU(h)

2,320,097 parameters. Tensor contract: (B, X, Y, Z, C_in) -> (B, X, Y, Z);
for the steady-state benchmark X = Y = 64, Z = 1, and C_in is the number of
physical input channels (3 for S2, 4 for S3, 7 for S4/S5).

The "FS" in the name is the training-side half of the design and lives in
train.py: a frequency-domain L1 term added to the spatial MSE. The two halves
are independent -- this file defines the operator, the loss decides how it is
supervised.

--------------------------------------------------------------------------
Design notes (run-to-run std on this benchmark is 0.086 K RMSE, n = 7):

  * U-Net branch applied to h rather than to the block input x_l. Against the
    standard U-FNO placement U(x_l), all five error metrics improve well
    outside noise (RMSE -15.5%, MAE -14.9%, MaxAE -9.3%, T_max -18.5%,
    Top-MAE -15.8%; Welch t = 3.5-4.8, p < 0.01, n = 6 vs n = 4). This was the
    only clearly super-noise effect across 42 training runs, so it is kept.

  * Axis-factorized spectral convolution: 3.11M -> 0.33M spectral parameters.
    No dense-spectral-convolution control was run, so its accuracy cost is
    unverified; do not claim "half the parameters at equal accuracy".

  * The backbone deliberately carries no local convolutional branch with a
    learned mixing scalar (-14,046 parameters) and no FiLM modulator driven by
    global features (-28,474 parameters). With both removed, all five metrics
    moved by |z| < 0.5 and run-to-run std halved (0.0864 -> 0.0438). Their
    contribution is *indistinguishable at this benchmark's variance*, which is
    why they are dropped for parsimony -- not a demonstration that they are
    useless.
--------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
#  Axis-factorized spectral convolution
#     A 3D spectral kernel is replaced by the sum of three 1D transforms
#     along X / Y / Z, each with a low-frequency and a high-frequency band.
#     Spectral parameters: 4*C^2*m1*m2*m3 -> 2*C^2*(m1+m2+m3).
# =========================================================================
class FactorizedSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.m1, self.m2, self.m3 = modes1, modes2, modes3
        s = 1.0 / (in_channels * out_channels)
        self.wx_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.wx_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat))
        self.wy_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.wy_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes2, dtype=torch.cfloat))
        self.wz_lo = nn.Parameter(s * torch.rand(in_channels, out_channels, modes3, dtype=torch.cfloat))
        self.wz_hi = nn.Parameter(s * torch.rand(in_channels, out_channels, modes3, dtype=torch.cfloat))

    def _axis(self, x, w_lo, w_hi, dim, modes):
        # x: (B, C, X, Y, Z). 1D rFFT along dim, multiply the low and high bands,
        # then inverse transform.
        xf = torch.fft.rfft(x, dim=dim)
        nd = xf.ndim
        perm = [0, 1, dim] + [d for d in range(2, nd) if d != dim]   # move the target axis to dim=2
        xf_p = xf.permute(*perm)
        Nf = xf_p.shape[2]
        rest = xf_p.shape[3:]
        B, C = x.shape[0], x.shape[1]
        R = 1
        for r in rest:
            R *= r
        xf_flat = xf_p.reshape(B, C, Nf, R)          # reshape: the permuted view is not contiguous
        out_flat = torch.zeros(B, self.out_channels, Nf, R, dtype=torch.cfloat, device=x.device)
        out_flat[:, :, :modes] = torch.einsum("bcmr,com->bomr", xf_flat[:, :, :modes], w_lo)
        if modes > 1:                                 # skip the high band when modes == 1 (the Z axis)
            out_flat[:, :, -modes:] = torch.einsum("bcmr,com->bomr", xf_flat[:, :, -modes:], w_hi)
        out_p = out_flat.reshape(B, self.out_channels, Nf, *rest)
        inv = [0] * nd
        for i, ax in enumerate(perm):
            inv[ax] = i
        out = out_p.permute(*inv)
        return torch.fft.irfft(out, n=x.shape[dim], dim=dim)

    def forward(self, x):
        return (self._axis(x, self.wx_lo, self.wx_hi, 2, self.m1) +
                self._axis(x, self.wy_lo, self.wy_hi, 3, self.m2) +
                self._axis(x, self.wz_lo, self.wz_hi, 4, self.m3))


# =========================================================================
#  U-Net branch used by the U-Fourier blocks (3, 4, 5)
# =========================================================================
class UNet3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = self._conv(input_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv2 = self._conv(input_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv2_1 = self._conv(input_channels, output_channels, kernel_size, 1, dropout_rate)
        self.conv3 = self._conv(input_channels, output_channels, kernel_size, 2, dropout_rate)
        self.conv3_1 = self._conv(input_channels, output_channels, kernel_size, 1, dropout_rate)
        self.deconv2 = self._deconv(input_channels, output_channels)
        self.deconv1 = self._deconv(input_channels * 2, output_channels)
        self.deconv0 = self._deconv(input_channels * 2, output_channels)
        self.output_layer = self._output(input_channels * 2, output_channels, kernel_size, 1, dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        return self.output_layer(concat0)

    def _conv(self, in_planes, out_ch, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_ch, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    def _deconv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def _output(self, in_ch, out_ch, kernel_size, stride, dropout_rate):
        return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                         padding=(kernel_size - 1) // 2)


# =========================================================================
#  Backbone: six blocks, the first three Fourier and the last three U-Fourier
# =========================================================================
class FSUFNOBackbone(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=3):
        super().__init__()
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        self.width = width
        self.in_channels = in_channels

        # Construction order (fc0 -> all conv -> all w -> unet -> fc1/fc2) is pinned:
        # it fixes how the RNG stream is consumed, so a given seed reproduces the
        # initial weights of the released checkpoints.
        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv4 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv5 = FactorizedSpectralConv3d(width, width, modes1, modes2, modes3)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.w4 = nn.Conv1d(width, width, 1)
        self.w5 = nn.Conv1d(width, width, 1)
        self.unet3 = UNet3d(width, width, 3, 0)
        self.unet4 = UNet3d(width, width, 3, 0)
        self.unet5 = UNet3d(width, width, 3, 0)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)                    # (B, width, X, Y, Z)

        def block(x, ci, wi, unet):
            x1 = ci(x)                                  # factorized spectral path K
            x2 = wi(x.reshape(batchsize, self.width, -1)).reshape(
                batchsize, self.width, size_x, size_y, size_z)
            x = x1 + x2                                 # h = K x + W x
            if unet is not None:
                x = x + unet(x)                         # U-Net applied to h, not to the block input
            return F.relu(x)

        x = block(x, self.conv0, self.w0, None)
        x = block(x, self.conv1, self.w1, None)
        x = block(x, self.conv2, self.w2, None)
        x = block(x, self.conv3, self.w3, self.unet3)
        x = block(x, self.conv4, self.w4, self.unet4)
        x = block(x, self.conv5, self.w5, self.unet5)

        x = x.permute(0, 2, 3, 4, 1)                    # (B, X, Y, Z, width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FSUFNO(nn.Module):
    """Outer wrapper: pad the input up to a multiple of 8, run the backbone,
    crop back to the original size.

    The U-Net branch downsamples three times with stride 2, so every spatial
    axis must be divisible by 8. For this task Z = 1 is replicate-padded to 8
    and modes3 is therefore min(modes, Z // 2 + 1) = 1.
    """
    def __init__(self, modes1=10, modes2=10, modes3=1, width=36, in_channels=3):
        super().__init__()
        self.block = FSUFNOBackbone(modes1, modes2, modes3, width, in_channels=in_channels)

    def forward(self, x):
        # x: (B, X, Y, Z, C_in)
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        px = (8 - size_x % 8) % 8
        py = (8 - size_y % 8) % 8
        pz = (8 - size_z % 8) % 8
        x = F.pad(x, (0, 0, 0, pz, 0, py), "replicate")
        if px:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, px), "constant", 0)
        x = self.block(x)
        x = x.view(batchsize, size_x + px, size_y + py, size_z + pz, 1)
        x = x[:, :size_x, :size_y, :size_z, :]
        return x.squeeze(-1)                            # (B, X, Y, Z)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
