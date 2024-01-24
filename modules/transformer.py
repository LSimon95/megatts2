import copy
from ctypes import Union

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from utils.utils import make_attn_mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, qkv_dim,  n_heads=8, dropout=0.):
        super().__init__()

        assert qkv_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = qkv_dim // n_heads
        self.dropout = dropout
        self.qkv_dim = qkv_dim

        self.w_q = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_k = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.w_v = nn.Linear(qkv_dim, qkv_dim, bias=True)

        self.out_proj = nn.Sequential(
            nn.Linear(qkv_dim, qkv_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv=None, mask=None):

        bsz, tgt_len, _ = q.size()
        src_len = kv.size(1) if kv is not None else tgt_len

        if kv is None:
            k = self.w_k(q)
            v = self.w_v(q)
            q = self.w_q(q)
        else:
            k = self.w_k(kv)
            v = self.w_v(kv)
            q = self.w_q(q)
        
        q = q.view(bsz, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            q, k, v, mask, self.dropout if self.training else 0.0, False)

        att = att.transpose(1, 2).contiguous().view(bsz, tgt_len, self.qkv_dim)

        return self.out_proj(att)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, conv_ff=False, n_heads=8, dropout=0.):
        super().__init__()

        self.dim = dim
        self.conv_ff = conv_ff
        self.n_heads = n_heads

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
                self.dropout,
                nn.Linear(ff_dim, dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):

        x = x + self.attn(self.norm1(x), mask=mask)
        if self.conv_ff:
            x = self.norm2(x)
            x = rearrange(x, 'B T D -> B D T')
            x = x + self.ff(x)
            x = rearrange(x, 'B D T -> B T D')
        else:
            x = x + self.ff(self.norm2(x))
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
        x_lens: torch.Tensor = None,
        causal: bool = False
    ) -> torch.Tensor:
        
        if x_lens is not None:
            mask = make_attn_mask(x_lens, self.layers[0].n_heads, causal=causal)
        else:
            mask = None
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


def test():

    x = torch.zeros([3, 7, 128]).to('cuda')
    context = torch.zeros([3, 20, 128]).to('cuda')

    x_lens = torch.Tensor([3, 4, 7]).to('cuda').to(torch.int32)
    context_lens = torch.Tensor([11, 12, 20]).to('cuda').to(torch.int32)

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


    context = encoder(context, x_lens=context_lens)
    print(context.shape)
