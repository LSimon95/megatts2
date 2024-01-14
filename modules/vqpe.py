import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.convnet import ConvNetDouble
from modules.tokenizer import (
    HIFIGAN_MEL_CHANNELS
)
from modules.quantization import ResidualVectorQuantizer

from einops import rearrange

class VQProsodyEncoder(nn.Module):
    def __init__(
            self,
            mel_bins: int = HIFIGAN_MEL_CHANNELS,
            stride:int = 8,
            hidden_size: int = 384,
            kernel_size: int = 5,
            n_stack: int = 3,
            n_block: int = 2,
            vq_bins: int = 1024,
            vq_dim: int = 256,
            activation: str = 'ReLU',
            ):
        super(VQProsodyEncoder, self).__init__()

        self.convnet = ConvNetDouble(
            in_channels=mel_bins,
            out_channels=vq_dim,
            hidden_size=hidden_size,
            n_stack=n_stack,
            n_block=n_block,
            middle_layer=nn.MaxPool1d(stride),
            kernel_size=kernel_size,
            activation=activation,
        )

        self.vq = ResidualVectorQuantizer(
            dimension=vq_dim,
            n_q=1,
            bins=vq_bins,
            decay=0.99
        )

    def forward(
            self, 
            mel: torch.Tensor, # (B, T, mel_bins)
            ):
        
        mel = rearrange(mel, "B T D -> B D T")
        ze = self.convnet(mel)
        zq, _, commit_loss = self.vq(ze)
        vq_loss = F.mse_loss(ze.detach(), zq)
        zq = rearrange(zq, "B D T -> B T D")
        return zq, commit_loss, vq_loss

def test():
    model = VQProsodyEncoder()
    mel = torch.randn(2, 303, 80)
    zq, commit_loss, vq_loss = model(mel)
    print(zq.shape, commit_loss.shape, vq_loss.shape)

