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
import glob
import librosa

from modules.tokenizer import TextTokenizer
from modules.feat_extractor import extract_mel_spec, VOCODER_SR, VOCODER_HOP_SIZE
from modules.datamodule import TokensCollector

import numpy as np
from modules.mrte import LengthRegulator
from speechbrain.pretrained import HIFIGAN

import torchaudio


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

            G_config = config['model']['G']

            mrte = instantiate_class(
                args=(), init=G_config['init_args']['mrte'])
            vqpe = instantiate_class(
                args=(), init=G_config['init_args']['vqpe'])

            G_config['init_args']['mrte'] = mrte
            G_config['init_args']['vqpe'] = vqpe

            G = instantiate_class(args=(), init=G_config)

            return G

    @classmethod
    def from_pretrained(self, ckpt: str, config: str) -> "MegaG":

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
            n_heads: int = 16,
            vq_dim: int = 512,
            tc_latent_dim: int = 512,
            vq_bins: int = 1024,
            dropout: float = 0.1,
    ):
        super(MegaPLM, self).__init__()
        d_model = vq_dim + tc_latent_dim
        self.plm = TransformerEncoder(
            TransformerEncoderLayer(
                dim=d_model,
                ff_dim=d_model * 4,
                n_heads=n_heads,
                dropout=dropout,
                conv_ff=False,
            ),
            num_layers=n_layers,
        )

        self.predict_layer = nn.Linear(d_model, vq_bins, bias=False)

        self.pos = SinePositionalEmbedding(d_model)
        self.pc_embedding = nn.Embedding(vq_bins + 2, vq_dim)

    def forward(
            self,
            tc_latent: torch.Tensor,  # (B, T, D)
            p_codes: torch.Tensor,  # (B, T)
            lens: torch.Tensor,  # (B,)
    ):
        pc_emb = self.pc_embedding(p_codes[:, :-1])
        x_emb = torch.cat([tc_latent, pc_emb], dim=-1)
        x_pos = self.pos(x_emb)

        x = self.plm(x_pos, lens, causal=True)
        logits = self.predict_layer(x)

        target = p_codes[:, 1:]

        return logits, target

    def infer(
            self,
            tc_latent: torch.Tensor,  # (B, T, D)
    ):
        T = tc_latent.shape[1]
        p_code = torch.Tensor([1024]).to(
            tc_latent.device).type(torch.int64).unsqueeze(0)
        for t in range(T):
            pc_emb = self.pc_embedding(p_code)
            x_emb = torch.cat([tc_latent[:, 0:t+1, :], pc_emb], dim=-1)
            x_pos = self.pos(x_emb)

            x = self.plm(x_pos)
            logits = self.predict_layer(x)[:, -1:, :]
            p_code = torch.cat([p_code, logits.argmax(dim=-1)], dim=1)

        return p_code[:, 1:]

    @classmethod
    def from_pretrained(cls, ckpt: str, config: str) -> "MegaPLM":

        with open(config, "r") as f:
            config = yaml.safe_load(f)

            plm_config = config['model']['plm']
            plm = instantiate_class(args=(), init=plm_config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('plm.'):
                state_dict[k[4:]] = v

        plm.load_state_dict(state_dict, strict=True)
        return plm


class MegaADM(nn.Module):
    def __init__(
            self,
            n_layers: int = 8,
            n_heads: int = 8,
            emb_dim: int = 256,
            tc_latent_dim: int = 512,
            tc_emb_dim: int = 256,
            dropout: float = 0.1,
            max_duration_token: int = 256,
    ):
        super(MegaADM, self).__init__()

        d_model = emb_dim + tc_emb_dim
        self.adm = TransformerEncoder(
            TransformerEncoderLayer(
                dim=d_model,
                ff_dim=emb_dim * 4,
                n_heads=n_heads,
                dropout=dropout,
                conv_ff=False,
            ),
            num_layers=n_layers,
        )

        self.dt_linear_emb = nn.Linear(1, emb_dim, bias=False)
        self.tc_linear_emb = nn.Linear(tc_latent_dim, tc_emb_dim, bias=False)
        self.pos_emb = SinePositionalEmbedding(d_model)
        self.predict_layer = nn.Linear(d_model, 1, bias=False)

        self.max_duration_token = max_duration_token

    def forward(
            self,
            tc_latents: torch.Tensor,  # (B, T, D)
            duration_tokens: torch.Tensor,  # (B, T)
            lens: torch.Tensor,  # (B,)
    ):
        dt_emb = self.dt_linear_emb(duration_tokens[:, :-1])
        tc_emb = self.tc_linear_emb(tc_latents)
        x_emb = torch.cat([tc_emb, dt_emb], dim=-1)
        x_pos = self.pos_emb(x_emb)

        x = self.adm(x_pos, lens, causal=True)
        duration_tokens_predict = self.predict_layer(x)[..., 0]

        target = duration_tokens[:, 1:, 0]

        # fill padding with 0
        # max_len = duration_tokens.size(1) - 1
        # seq_range = torch.arange(0, max_len, device=duration_tokens_predict.device)
        # expaned_lengths = seq_range.unsqueeze(0).expand(lens.size(0), max_len)
        # mask = expaned_lengths >= lens.unsqueeze(-1)
        # duration_tokens_predict = duration_tokens_predict.masked_fill(mask, 0)
        return duration_tokens_predict, target

    def infer(
        self,
        tc_latents: torch.Tensor,  # (B, T, D)
    ):
        T = tc_latents.shape[1]
        p_code = torch.Tensor([0]).to(
            tc_latents.device).unsqueeze(0).unsqueeze(1)
        for t in range(T):
            dt_emb = self.dt_linear_emb(p_code)
            tc_emb = self.tc_linear_emb(tc_latents[:, 0:t+1, :])

            x_emb = torch.cat([tc_emb, dt_emb], dim=-1)
            x_pos = self.pos_emb(x_emb)

            x = self.adm(x_pos)
            dt_predict = self.predict_layer(x)[:, -1:, :]
            p_code = torch.cat([p_code, dt_predict], dim=1)

        return (p_code[:, 1:, :] + 0.5).to(torch.int32).clamp(1, 128)

    @classmethod
    def from_pretrained(self, ckpt: str, config: str) -> "MegaADM":

        with open(config, "r") as f:
            config = yaml.safe_load(f)

            adm_config = config['model']['adm']
            adm = instantiate_class(args=(), init=adm_config)

        state_dict = {}
        for k, v in torch.load(ckpt)['state_dict'].items():
            if k.startswith('adm.'):
                state_dict[k[4:]] = v

        adm.load_state_dict(state_dict, strict=True)
        return adm


class Megatts(nn.Module):
    def __init__(
        self,
        g_ckpt: str,
        g_config: str,
        plm_ckpt: str,
        plm_config: str,
        adm_ckpt: str,
        adm_config: str,
        symbol_table: str
    ):
        super(Megatts, self).__init__()

        self.generator = MegaG.from_pretrained(g_ckpt, g_config)
        self.generator.eval()
        self.plm = MegaPLM.from_pretrained(plm_ckpt, plm_config)
        self.plm.eval()
        self.adm = MegaADM.from_pretrained(adm_ckpt, adm_config)
        self.adm.eval()

        self.tt = TextTokenizer()
        self.ttc = TokensCollector(symbol_table)

        self.lr = LengthRegulator(
            VOCODER_HOP_SIZE, VOCODER_SR, (VOCODER_HOP_SIZE / VOCODER_SR * 1000))

        self.hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-libritts-16kHz")
        self.hifi_gan.eval()

    def forward(
            self,
            wavs_dir: str,
            text: str,
    ):
        mels_prompt = None
        # Make mrte mels
        wavs = glob.glob(f'{wavs_dir}/*.wav')
        mels = torch.empty(0)
        for wav in wavs:
            y = librosa.load(wav, sr=VOCODER_SR)[0]
            y = librosa.util.normalize(y)
            # y = librosa.effects.trim(y, top_db=20)[0]
            y = torch.from_numpy(y)

            mel_spec = extract_mel_spec(y).transpose(0, 1)
            mels = torch.cat([mels, mel_spec], dim=0)

            if mels_prompt is None:
                mels_prompt = mel_spec

        mels = mels.unsqueeze(0)

        # G2P
        phone_tokens = self.ttc.phone2token(
            self.tt.tokenize_lty(self.tt.tokenize(text)))
        phone_tokens = phone_tokens.unsqueeze(0)

        with torch.no_grad():
            tc_latent = self.generator.mrte.tc_latent(phone_tokens, mels)
            dt = self.adm.infer(tc_latent)[..., 0]
            tc_latent_expand = self.lr(tc_latent, dt)
            tc_latent = F.max_pool1d(tc_latent_expand.transpose(
                1, 2), 8, ceil_mode=True).transpose(1, 2)
            p_codes = self.plm.infer(tc_latent)

            zq = self.generator.vqpe.vq.decode(p_codes.unsqueeze(0))
            zq = rearrange(
                zq, "B D T -> B T D").unsqueeze(2).contiguous().expand(-1, -1, 8, -1)
            zq = rearrange(zq, "B T S D -> B (T S) D")
            x = torch.cat(
                [tc_latent_expand, zq[:, :tc_latent_expand.shape[1], :]], dim=-1)
            x = rearrange(x, 'B T D -> B D T')
            x = self.generator.decoder(x)

            audio = self.hifi_gan.decode_batch(x.cpu())
            audio_prompt = self.hifi_gan.decode_batch(
                mels_prompt.unsqueeze(0).transpose(1, 2).cpu())
            audio = torch.cat([audio_prompt, audio], dim=-1)

            torchaudio.save('test.wav', audio[0], VOCODER_SR)
