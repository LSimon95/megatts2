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

        
        q = q.view(bsz, self.n_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.n_heads, src_len, self.head_dim)
        v = v.view(bsz, self.n_heads, src_len, self.head_dim)

        att = F.scaled_dot_product_attention(
            q, k, v, mask, self.dropout, False)
        att = att.permute(2, 0, 1, 3).contiguous().view(bsz, tgt_len, self.qkv_dim)

        return self.out_proj(att)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, ff_dim, conv_ff=False, n_heads=8, dropout=0.):
        super().__init__()

        self.dim = dim
        self.query_dim = dim
        self.conv_ff = conv_ff
        self.n_heads = n_heads

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

        x = x + self.attn1(self.norm1(x), mask=mask)
        x = x + self.attn2(self.norm2(x), kv=context)
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
                nn.Linear(ff_dim, dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):

        x = x + self.attn(self.norm1(x), mask=mask)
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
        x_lens: torch.Tensor,
    ) -> torch.Tensor:

        mask = make_attn_mask(x_lens, self.layers[0].n_heads)

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
        x_lens: torch.Tensor,
    ) -> torch.Tensor:

        mask = make_attn_mask(x_lens, self.layers[0].n_heads)

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

    context = encoder(context, x_lens=context_lens)
    print(context.shape)
    out = decoder(x, context, x_lens=x_lens)
    print(out.shape)