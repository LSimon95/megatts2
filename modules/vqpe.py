import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.convnet import ConvNet
from modules.tokenizer import (
    HIFIGAN_MEL_CHANNELS
)
from modules.quantization import ResidualVectorQuantizer

from einops import rearrange

from typing import List

class VQProsodyEncoder(nn.Module):
    def __init__(
            self,
            hidden_sizes: List = [HIFIGAN_MEL_CHANNELS, 256, 256, 512, 512],
            kernel_size: int = 5,
            stack_size: int = 3,
            activation: str = 'ReLU',
            ):
        super(VQProsodyEncoder, self).__init__()

        self.convnet = ConvNet(
            hidden_sizes=hidden_sizes,
            kernel_size=kernel_size,
            stack_size=stack_size,
            activation=activation,
        )

        self.vq = ResidualVectorQuantizer(
            dimension=512,
            n_q=1,
            bins=1024,
            decay=0.99
        )

    def forward(
            self, 
            mel: torch.Tensor, # (B, T, mel_bins)
            ):
        
        mel = rearrange(mel, "B T D -> B D T")
        ze = self.convnet(mel)
        # zq, _, commit_loss = self.vq(ze)
        # vq_loss = F.mse_loss(ze.detach(), zq)
        # zq = rearrange(zq, "B D T -> B T D")
        return ze.transpose(1, 2), None, None

def test():
    model = VQProsodyEncoder()
    mel = torch.randn(2, 256, 80)
    zq, commit_loss = model(mel)
    print(zq.shape, commit_loss.shape)

