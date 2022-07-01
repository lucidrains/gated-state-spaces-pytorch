import torch
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange

# classes

class DSS(nn.Module):
    def __init__(
        self,
        *,
        dim,
        kernel_N = 512
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Lambda

        self.Lambda_real = nn.Parameter(torch.randn(kernel_N))
        self.Lambda_imag = nn.Parameter(torch.randn(kernel_N))

        # C

        self.C_real = nn.Parameter(torch.randn(dim, kernel_N))
        self.C_imag = nn.Parameter(torch.randn(dim, kernel_N))

        # params D

        self.param_D = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        """
        einstein notation:
        b - batch
        l - sequence length
        d - dimension
        """

        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # learned weighted residual

        residual = u * self.param_D

        # derive simple dss kernel

        Lambda = -self.Lambda_real.exp() + 1j * self.Lambda_imag.exp()
        C = self.C_real + 1j * self.C_imag

        arange = torch.arange(seq_len, device = device)

        S = (rearrange(Lambda, 'n -> n 1') * rearrange(arange, 'l -> 1 l')).exp()
        C = C * (Lambda.exp() - 1) / Lambda

        K = einsum('h n, n l -> h l', C, S).real

        # conv1d fft O(nlog(n))

        u = rearrange(u, 'b l d -> b d l')

        u_f = rfft(u, n = seq_len * 2, dim = -1)
        K_f = rfft(K, n = seq_len * 2, dim = -1)

        y = irfft(u_f * K_f, seq_len * 2, dim = -1)[..., :seq_len]
        y = rearrange(y, 'b d l -> b l d')

        return y + residual

class GSS(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dss_kernel_H, bias = False), nn.GELU())

        self.dss = DSS(dim = dss_kernel_H, kernel_N = dss_kernel_N)

        self.to_gate = nn.Linear(dss_kernel_H, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dss(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        return out + residual
