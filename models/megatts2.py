import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.mrte import MRTE
from modules.vqpe import VQProsodyEncoder
from modules.convnet import ConvNet
from modules.embedding import SinePositionalEmbedding, TokenEmbedding

from modules.transformer import TransformerEncoder, TransformerEncoderLayer

from einops import rearrange

import yaml

from utils.utils import instantiate_class


class MegaG(nn.Module):
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
        super(MegaG, self).__init__()

        self.mrte = mrte
        self.vqpe = vqpe
        self.decoder = ConvNet(
            in_channels=mrte.hidden_size + vqpe.vq.dimension,
            out_channels=mrte.mel_bins,
            hidden_size=hidden_size,
            n_stacks=decoder_n_stack,
            n_blocks=decoder_n_block,
            kernel_size=kernel_size,
            activation=activation,
        )

    def forward(
            self,
            duration_tokens: torch.Tensor,  # (B, T)
            phone: torch.Tensor,  # (B, T)
            phone_lens: torch.Tensor,  # (B,)
            mel_mrte: torch.Tensor,  # (B, T, mel_bins)
            mel_vqpe: torch.Tensor,  # (B, T, mel_bins)
    ):
        zq, commit_loss, vq_loss, _ = self.vqpe(mel_vqpe)
        x = self.mrte(duration_tokens, phone, phone_lens, mel_mrte)

        x = torch.cat([x, zq], dim=-1)

        x = rearrange(x, 'B T D -> B D T')
        x = self.decoder(x)
        x = rearrange(x, 'B D T -> B T D')

        return x, commit_loss, vq_loss

    def s2_latent(
            self,
            phone: torch.Tensor,  # (B, T)
            phone_lens: torch.Tensor,  # (B,)
            mel_mrte: torch.Tensor,  # (B, T, mel_bins)
            mel_vqpe: torch.Tensor,  # (B, T, mel_bins)
    ):
        _, _, _, codes = self.vqpe(mel_vqpe)
        x = self.mrte.tc_latent(phone, phone_lens, mel_mrte)
        return x, codes
    
    @classmethod
    def from_hparams(self, config_path: str) -> "MegaG":

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

            G_config = init=config['model']['G']

            mrte = instantiate_class(args=(), init=G_config['init_args']['mrte'])
            vqpe = instantiate_class(args=(), init=G_config['init_args']['vqpe'])

            G_config['init_args']['mrte'] = mrte
            G_config['init_args']['vqpe'] = vqpe

            G = instantiate_class(args=(), init=G_config)

            return G
    
    @classmethod
    def from_pretrained(self, ckpt: str, config : str) -> "MegaG":

        G = MegaG.from_hparams(config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('G.'):
                state_dict[k[2:]] = v

        G.load_state_dict(state_dict, strict=True)
        return G

class MegaPLM(nn.Module):
    def __init__(
            self,
            n_layers: int = 12,
            n_heads: int = 6,
            d_model: int = 1026,
            hidden_size: int = 1024,
            ff_dim: int = 1024,
            vq_bins: int = 1024
    ):
        super(MegaPLM, self).__init__()

        self.plm = TransformerEncoder(
            TransformerEncoderLayer(
                dim=d_model,
                ff_dim=ff_dim,
                conv_ff=False,
                n_heads=n_heads,
            ),
            num_layers=n_layers,
        )

        self.predict_layer = nn.Linear(d_model, hidden_size)

        self.pos = SinePositionalEmbedding(hidden_size)
        self.pc_embedding = nn.Embedding(vq_bins + 2, hidden_size)

        self.max_pool = nn.MaxPool1d(2)

    def forward(
            self,

            tc_latent: torch.Tensor,  # (B, T, D)
            x_lens: torch.Tensor,  # (B,)
    ):
        x = self.plm(x, x_lens)
        x = self.predict_layer(x)
        return x