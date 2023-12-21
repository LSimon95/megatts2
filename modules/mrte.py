import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import List

from modules.embedding import TokenEmbedding, SinePositionalEmbedding
from modules.convnet import ConvNet
from modules.transformer import (TransformerEncoder, 
                                 TransformerEncoderLayer,
                                 TransformerDecoder,
                                 TransformerDecoderLayer)

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

    def __init__(self, mel_frames, sample_rate, duration_tokne_ms):
        super(LengthRegulator, self).__init__()

        assert (mel_frames / sample_rate * 1000 / duration_tokne_ms) == 1

    def forward(
                self,
                x: torch.Tensor, # (B, T, D)
                duration_tokens: torch.Tensor, # (B, T) int for duration, unit is 10ms
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
            attn_dim: int = 512,
            ff_dim: int = 1024,
            nhead: int = 2,
            n_layers: int = 8,
            ge_kernel_size: int = 31,
            ge_hidden_sizes: List = [HIFIGAN_MEL_CHANNELS, 256, 256, 512, 512],
            ge_activation: str = 'ReLU',
            ge_out_channels: int = 512,
            duration_tokne_ms: int = (HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000),
            text_vocab_size: int = 320,
            dropout: float = 0.1,
            sample_rate: int = HIFIGAN_SR,
    ):
        super(MRTE, self).__init__()
        
        self.text_embedding = TokenEmbedding(
            dim_model=attn_dim,
            vocab_size=text_vocab_size,
            dropout=dropout,
        )

        self.text_pos_embedding = SinePositionalEmbedding(
            dim_model=attn_dim,
            dropout=dropout,
        )

        self.mel_embedding = nn.Linear(mel_bins, attn_dim)
        self.mel_pos_embedding = SinePositionalEmbedding(
            dim_model=attn_dim,
            dropout=dropout,
        )

        self.mel_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                dim=attn_dim,
                ff_dim=ff_dim,
                conv_ff=True,
                n_heads=nhead,
                dropout=dropout,
            ),
            num_layers=n_layers,
        )

        self.mrte_decoder = TransformerDecoder(
            TransformerDecoderLayer(
                dim=attn_dim,
                ff_dim=ff_dim,
                n_heads=nhead,
                dropout=dropout,
            ),
            num_layers=n_layers,
        )

        self.compress_features = nn.Linear(attn_dim + ge_out_channels, ge_out_channels)

        self.ge = ConvNet(
            hidden_sizes = ge_hidden_sizes,
            kernel_size = ge_kernel_size,
            stack_size  = 3,
            activation = ge_activation,
            avg_pooling = True
        )

        self.length_regulator = LengthRegulator(mel_frames, sample_rate, duration_tokne_ms)

    def forward(
            self,
            duration_tokens: torch.Tensor, # (B, T)
            text: torch.Tensor, # (B, T)
            text_lens: torch.Tensor, # (B,)
            mel: torch.Tensor, # (B, T, mel_bins)
            mel_lens: torch.Tensor, # (B,)
    ):
        
        text = self.text_embedding(text)
        text = self.text_pos_embedding(text)

        mel_emb = self.mel_embedding(mel)
        mel_pos = self.mel_pos_embedding(mel_emb)

        mel_context = self.mel_encoder(mel_pos, mel_lens)
        phone = self.mrte_decoder(text, mel_context, text_lens)

        mel = rearrange(mel, "B T D -> B D T")
        ge = self.ge(mel)
        ge = ge.unsqueeze(1).repeat(1, phone.shape[1], 1)

        out = self.compress_features(torch.cat([ge, phone], dim=-1))
        out = self.length_regulator(out, duration_tokens)

        return out

def test():
    lr_in = torch.randn(2, 10, 128)
    lr = LengthRegulator(240, 16000, 15)

    duration_tokens = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 5]]).to(dtype=torch.int32)

    out = lr(lr_in, duration_tokens)
    assert out.shape == (2, 11, 128)

    mrte = MRTE(
        mel_bins = HIFIGAN_MEL_CHANNELS,
        mel_frames = HIFIGAN_HOP_LENGTH,
        attn_dim = 512,
        ff_dim = 1024,
        nhead = 2,
        n_layers = 8,
        ge_kernel_size = 31,
        ge_hidden_sizes = [HIFIGAN_MEL_CHANNELS, 256, 256, 512, 512],
        ge_activation = 'ReLU',
        ge_out_channels = 512,
        duration_tokne_ms = (HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000),
        text_vocab_size = 320,
        dropout = 0.1,
        sample_rate = HIFIGAN_SR,
    )
    mrte = mrte.to('cuda')

    duration_tokens = torch.tensor([[1, 2, 3, 4], [1, 1, 1, 2]]).to(dtype=torch.int32).to('cuda')

    t = torch.randint(0, 320, (2, 10)).to(dtype=torch.int64).to('cuda')
    tl = torch.tensor([6, 10]).to(dtype=torch.int64).to('cuda')
    m = torch.randn(2, 2400, HIFIGAN_MEL_CHANNELS).to('cuda')
    ml = torch.tensor([1200, 2400]).to(dtype=torch.int64).to('cuda')

    out = mrte(duration_tokens, t, tl, m, ml)
    print(out.shape)

