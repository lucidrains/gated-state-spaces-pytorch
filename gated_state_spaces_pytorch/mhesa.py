import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

def exists(val):
    return val is not None

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# classes

class MHESA(nn.Module):
    """ used for time-series in ETSFormer https://arxiv.org/abs/2202.01381 """

    def __init__(
        self,
        *,
        dim,
        heads
    ):
        super().__init__()
        assert (dim % heads) == 0

        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.alphas = nn.Parameter(torch.randn(heads))

        # params D

        self.param_D = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        """
        einstein notation:
        b - batch
        h - heads
        l - sequence length
        d - dimension
        """

        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # learned weighted residual

        residual = u * self.param_D

        # weights derived from alphas (learned exponential smoothing decay rate)

        alphas = self.alphas.sigmoid()
        reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
        K = alphas * ((1 - alphas) ** rearrange(reversed_powers, '... l -> ... l 1'))

        # conv1d fft O(nlog(n))

        u = rearrange(u, '... (h d) -> ... h d', h = self.heads)

        out = conv1d_fft(u, K, dim = -3, weight_dim = -2)

        out = rearrange(out, '... h d -> ... (h d)')

        return out + residual

class GatedMHESA(nn.Module):
    """ Pseudocode 3.2 """
    """ except state spaces replaced with multi-head exponential smoothing with learned alpha """
    """ used for time-series in ETSFormer https://arxiv.org/abs/2202.01381 """

    def __init__(
        self,
        *,
        dim,    
        heads = 8,
        dim_mhesa = 512,
        dim_expansion_factor = 4,
    ):
        super().__init__()
        assert (dim_mhesa % heads) == 0

        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dim_mhesa, bias = False), nn.GELU())

        self.mhesa = MHESA(dim = dim_mhesa, heads = heads)

        self.to_gate = nn.Linear(dim_mhesa, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.mhesa(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        return out + residual

# Gated Dsconv LM

class GatedExponentialSmoothingLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_mhesa = 512,
        dim_expansion_factor = 4,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                GatedMHESA(
                    dim = dim,
                    heads = heads,
                    dim_mhesa = dim_mhesa,
                    dim_expansion_factor = dim_expansion_factor
                )
            )

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, labels = None):
        x = self.token_emb(x)

        for mhesa in self.layers:
            x = mhesa(x)

        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
