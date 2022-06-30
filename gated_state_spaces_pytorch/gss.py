import torch
from torch import nn, einsum

from einops import rearrange, repeat, reduce

class GSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
