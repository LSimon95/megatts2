import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mrte import MRTE
from modules.vqpe import VQProsodyEncoder
from modules.convnet import ConvNet
from modules.embedding import SinePositionalEmbedding, TokenEmbedding

from modules.transformer import TransformerEncoder, TransformerEncoderLayer

from einops import rearrange

class MegaVQ(nn.Module):
    def __init__(
            self,
            mrte: MRTE,
            vqpe: VQProsodyEncoder,
            kernel_size: int = 5,
            activation: str = 'ReLU',
            hidden_size: int = 512,
            decoder_n_stack: int = 4,
            decoder_n_block: int = 2

    ):
        super(MegaVQ, self).__init__()

        self.mrte = mrte
        self.vqpe = vqpe
        self.decoder = ConvNet(
            in_channels=mrte.hidden_size + vqpe.vq.dimension,
            out_channels=mrte.mel_bins,
            hidden_size=hidden_size,
            n_stack=decoder_n_stack,
            n_block=decoder_n_block,
            kernel_size=kernel_size,
            activation=activation,
        )

    def forward(
            self,
            duration_tokens: torch.Tensor,  # (B, T)
            text: torch.Tensor,  # (B, T)
            text_lens: torch.Tensor,  # (B,)
            mel_mrte: torch.Tensor,  # (B, T, mel_bins)
            mel_lens_mrte: torch.Tensor,  # (B,)
            mel_vqpe: torch.Tensor,  # (B, T, mel_bins)
    ):
        zq, commit_loss, vq_loss = self.vqpe(mel_vqpe)
        x = self.mrte(duration_tokens, text, text_lens,
                      mel_mrte, mel_lens_mrte)
        x = torch.cat([x, zq], dim=-1)

        x = rearrange(x, 'B T D -> B D T')
        x = self.decoder(x)
        x = rearrange(x, 'B D T -> B T D')

        return x, commit_loss, vq_loss