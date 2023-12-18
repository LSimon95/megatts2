import copy
from ctypes import Union

import torch
from torch import nn
import xformers.ops as xops

from einops import rearrange


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, qkv_dim,  n_heads=8, dropout=0.):
        super().__init__()

        assert qkv_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = qkv_dim // n_heads
        self.dropout = dropout

        self.w_q = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_k = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_v = nn.Linear(qkv_dim, qkv_dim, bias=True)

        self.out_proj = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv=None, mask=None):

        if kv is None:            
            k = self.w_k(q)
            v = self.w_v(q)
            q = self.w_q(q)
        else:
            k = self.w_k(kv)
            v = self.w_v(kv)
            q = self.w_q(q)


        q = rearrange(q, 'b t (h d) -> b t h d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.n_heads)

        att = xops.memory_efficient_attention(
            q, k, v, attn_bias=mask, p=self.dropout)

        att = rearrange(att, 'b t h d -> b t (h d)')
        return self.out_proj(att)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, conv_ff=False, n_heads=8, dropout=0.):
        super().__init__()

        self.dim = dim
        self.query_dim = dim
        self.conv_ff = conv_ff

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.attn1 = MultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)
        self.attn2 = MultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        if conv_ff:
            self.ff = nn.Sequential(
                nn.Conv1d(dim, ff_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(ff_dim, dim, kernel_size=5, padding=2),
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, dim),
            )

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            mask: torch.Tensor = None,
    ):

        x = x + self.attn1(self.norm1(x), mask)
        x = x + self.attn2(self.norm2(x), context)
        if self.conv_ff:
            x = x + self.ff(self.norm3(x).transpose(1, 2)).transpose(1, 2)
        else:
            x = x + self.ff(self.norm3(x))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, conv_ff=False, n_heads=8, dropout=0.):
        super().__init__()

        self.dim = dim
        self.conv_ff = conv_ff

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = MultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        if conv_ff:
            self.ff = nn.Sequential(
                nn.Conv1d(dim, ff_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(ff_dim, dim, kernel_size=5, padding=2),
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):

        x = x + self.attn(self.norm1(x), mask)
        if self.conv_ff:
            x = x + self.ff(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        else:
            x = x + self.ff(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer: TransformerDecoderLayer,
            num_layers: int,
            norm=None
    ):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, context, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            num_layers: int,
            norm=None
    ):
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


def test():

    x = torch.zeros([3, 7, 128]).to('cuda')
    context = torch.zeros([3, 20, 128]).to('cuda')

    encoder = TransformerEncoder(
        TransformerEncoderLayer(
            128,
            4 * 128,
            n_heads=8,
            dropout=0.1,
            conv_ff=True
        ),
        12,
        nn.LayerNorm(128)
    ).to('cuda')

    decoder = TransformerDecoder(
        TransformerDecoderLayer(
            128,
            4 * 128,
            n_heads=8,
            dropout=0.1,
            conv_ff=True
        ),
        12,
        nn.LayerNorm(128)
    ).to('cuda')

    context = encoder(context,)
    print(context.shape)
    out = decoder(x, context,)
    print(out.shape)
