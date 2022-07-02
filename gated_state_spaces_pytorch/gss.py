import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange

# functions

def exists(val):
    return val is not None

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

        K = einsum('h n, n l -> l h', C, S).real

        # conv1d fft O(nlog(n))

        u_f = rfft(u, n = seq_len * 2, dim = -2)
        K_f = rfft(K, n = seq_len * 2, dim = -2)

        y = irfft(u_f * K_f, seq_len * 2, dim = -2)[..., :seq_len, :]

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

# Gated State Spaces LM

class GatedStateSpacesLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                GSS(
                    dim = dim,
                    dss_kernel_H = dss_kernel_H,
                    dss_kernel_N = dss_kernel_N,
                    dim_expansion_factor = dim_expansion_factor
                )
            ]))

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, labels = None):
        x = self.token_emb(x)

        for gss, in self.layers:
            x = gss(x)

        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
