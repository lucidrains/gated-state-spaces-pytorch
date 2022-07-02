import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange

# functions

def exists(val):
    return val is not None

# classes

class EfficientDsConv(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        max_seq_len
    ):
        super().__init__()
        assert (dim % heads) == 0

        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.weight = nn.Parameter(torch.randn(max_seq_len, heads))

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

        # dsconv kernel depends on sequence length

        K = self.weight[-seq_len:]

        # conv1d fft O(nlog(n))

        u_f = rfft(u, n = seq_len * 2, dim = -2)
        K_f = rfft(K, n = seq_len * 2, dim = -2)

        u_f = rearrange(u_f, '... (h d) -> ... h d', h = self.heads)
        K_f = rearrange(K_f, '... -> ... 1')

        out = rearrange(u_f * K_f, '... h d -> ... (h d)')

        out = irfft(out, seq_len * 2, dim = -2)[..., :seq_len, :]

        return out + residual

class GatedDsConv(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        max_seq_len,
        heads = 8,
        dim_dsconv = 512,
        dim_expansion_factor = 4,
    ):
        super().__init__()
        assert (dim_dsconv % heads) == 0

        self.norm = nn.LayerNorm(dim)
        self.max_seq_len = max_seq_len

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dim_dsconv, bias = False), nn.GELU())

        self.dsconv = EfficientDsConv(dim = dim_dsconv, heads = heads, max_seq_len = max_seq_len)

        self.to_gate = nn.Linear(dim_dsconv, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        assert x.shape[1] <= self.max_seq_len

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dsconv(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        return out + residual

# Gated State Spaces LM

class GatedDsConvLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_dsconv = 512,
        max_seq_len = 2048,
        dim_expansion_factor = 4,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                GatedDsConv(
                    dim = dim,
                    heads = heads,
                    max_seq_len = max_seq_len,
                    dim_dsconv = dim_dsconv,
                    dim_expansion_factor = dim_expansion_factor
                )
            )

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, labels = None):
        assert x.shape[1] <= self.max_seq_len

        x = self.token_emb(x)

        for gss in self.layers:
            x = gss(x)

        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
