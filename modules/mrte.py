import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import List

from modules.embedding import TokenEmbedding, SinePositionalEmbedding
from modules.convnet import ConvNetDouble, ConvNet
from modules.transformer import (TransformerEncoder,
                                 TransformerEncoderLayer,
                                 MultiHeadAttention)
from utils.utils import make_attn_mask

from modules.tokenizer import (
    HIFIGAN_SR,
    HIFIGAN_MEL_CHANNELS,
    HIFIGAN_HOP_LENGTH
)


def create_alignment(base_mat, duration_tokens):
    N, L = duration_tokens.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_tokens[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_tokens[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator from FastSpeech """

    def __init__(self, mel_frames, sample_rate, duration_token_ms):
        super(LengthRegulator, self).__init__()

        assert (mel_frames / sample_rate * 1000 / duration_token_ms) == 1

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        duration_tokens: torch.Tensor,  # (B, T) int for duration
        mel_max_length=None
    ):

        bsz, input_len, _ = x.size()

        expand_max_len = torch.max(torch.sum(duration_tokens, -1), -1)[0].int()

        alignment = torch.zeros(bsz, expand_max_len, input_len).numpy()
        alignment = create_alignment(alignment, duration_tokens.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)
        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output


class MRTE(nn.Module):
    def __init__(
            self,
            mel_bins: int = HIFIGAN_MEL_CHANNELS,
            mel_frames: int = HIFIGAN_HOP_LENGTH,
            mel_activation: str = 'ReLU',
            mel_kernel_size: int = 3,
            mel_stride: int = 16,
            mel_n_layer: int = 5,
            mel_n_stack: int = 5,
            mel_n_block: int = 2,
            content_ff_dim: int = 1024,
            content_n_heads: int = 2,
            content_n_layers: int = 8,
            hidden_size: int = 512,
            duration_token_ms: float = (
                HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000),
            phone_vocab_size: int = 320,
            dropout: float = 0.1,
            sample_rate: int = HIFIGAN_SR,
    ):
        super(MRTE, self).__init__()

        self.n_heads = content_n_heads
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size

        self.phone_embedding = TokenEmbedding(
            dim_model=hidden_size,
            vocab_size=phone_vocab_size,
            dropout=dropout,
        )

        self.phone_pos_embedding = SinePositionalEmbedding(
            dim_model=hidden_size,
            dropout=dropout,
        )

        self.mel_encoder_middle_layer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=mel_stride + 1,
            stride=mel_stride,
            padding=(mel_stride) // 2,
        )
        self.mel_encoder = ConvNetDouble(
            in_channels=mel_bins,
            out_channels=hidden_size,
            hidden_size=hidden_size,
            n_layers=mel_n_layer,
            n_stacks=mel_n_stack,
            n_blocks=mel_n_block,
            middle_layer=self.mel_encoder_middle_layer,
            kernel_size=mel_kernel_size,
            activation=mel_activation,
        )

        self.phone_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=hidden_size,
                ff_dim=content_ff_dim,
                conv_ff=True,
                n_heads=content_n_heads,
                dropout=dropout,
            ),
            num_layers=content_n_layers,
        )

        self.mha = MultiHeadAttention(
            qkv_dim=hidden_size,
            n_heads=1,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()

        self.length_regulator = LengthRegulator(
            mel_frames, sample_rate, duration_token_ms)
        

        # self.test_pllm = TransformerEncoder(
        #     TransformerEncoderLayer(
        #         dim=1024,
        #         ff_dim=1024,
        #         conv_ff=True,
        #         n_heads=16,
        #         dropout=dropout,
        #     ),
        #     num_layers=12,
        # )

    def tc_latent(
            self,
            phone: torch.Tensor,  # (B, T)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        phone_emb = self.phone_embedding(phone)
        phone_pos = self.phone_pos_embedding(phone_emb)

        mel = rearrange(mel, 'B T D -> B D T')
        mel_context = self.mel_encoder(mel)
        mel_context = rearrange(mel_context, 'B D T -> B T D')
        phone_x = self.phone_encoder(phone_pos)

        tc_latent = self.mha(phone_x, kv=mel_context)
        tc_latent = self.norm(tc_latent)
        tc_latent = self.activation(tc_latent)

        return tc_latent

    def forward(
            self,
            duration_tokens: torch.Tensor,  # (B, T)
            phone: torch.Tensor,  # (B, T)
            phone_lens: torch.Tensor,  # (B,)
            mel: torch.Tensor,  # (B, T, mel_bins)
    ):
        tc_latent = self.tc_latent(phone, phone_lens, mel)
        
        out = self.length_regulator(tc_latent, duration_tokens)
        return out


def test():
    lr_in = torch.randn(2, 10, 128)
    lr = LengthRegulator(240, 16000, 15)

    duration_tokens = torch.tensor(
        [[1, 2, 3, 4], [1, 2, 3, 5]]).to(dtype=torch.int32)

    out = lr(lr_in, duration_tokens)
    assert out.shape == (2, 11, 128)

    mrte = MRTE(
        mel_bins = HIFIGAN_MEL_CHANNELS,
        mel_frames = HIFIGAN_HOP_LENGTH,
        ff_dim = 1024,
        n_heads = 2,
        n_layers = 8,
        hidden_size = 512,
        activation = 'ReLU',
        kernel_size = 3,
        stride = 16,
        n_stacks = 5,
        n_blocks = 2,
        duration_token_ms = (
            HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000),
        phone_vocab_size = 320,
        dropout = 0.1,
        sample_rate = HIFIGAN_SR,
    )
    mrte = mrte.to('cuda')

    duration_tokens = torch.tensor([[1, 2, 3, 4], [1, 1, 1, 2]]).to(
        dtype=torch.int32).to('cuda')

    t = torch.randint(0, 320, (2, 10)).to(dtype=torch.int64).to('cuda')
    tl = torch.tensor([6, 10]).to(dtype=torch.int64).to('cuda')
    m = torch.randn(2, 347, HIFIGAN_MEL_CHANNELS).to('cuda')

    out = mrte(duration_tokens, t, tl, m)
