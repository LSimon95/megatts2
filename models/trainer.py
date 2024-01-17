import lightning.pytorch as pl

import torch
import torchaudio
import torch.nn.functional as F

import transformers

import numpy as np
import math

from .megatts2 import MegaG
from modules.dscrm import Discriminator
from modules.tokenizer import HIFIGAN_SR

from utils.utils import plot_spectrogram_to_numpy

from speechbrain.pretrained import HIFIGAN

class MegaGANTrainer(pl.LightningModule):
    def __init__(
            self,
            G: MegaG,
            D: Discriminator,
            initial_learning_rate: float,
            warmup_steps: float = 200,
            G_commit_loss_coeff: float = 10,
            G_vq_loss_coeff: float = 10,
            G_adv_loss_coeff: float = 1.0,

            train_dtype: str = "float32",
            **kwargs
    ):

        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['G', 'D'])
        self.G = G
        self.D = D
        self.validation_step_outputs = []

        if self.hparams.train_dtype == "float32":
            self.train_dtype = torch.float32
        elif self.hparams.train_dtype == "bfloat16":
            self.train_dtype = torch.bfloat16
            print("Using bfloat16")

    def configure_optimizers(self):
        D_params = [
            {"params": self.D.parameters()}
        ]
        G_params = [
            {"params": self.G.parameters()}
        ]

        D_opt = torch.optim.AdamW(
            D_params, lr=self.hparams.initial_learning_rate)
        G_opt = torch.optim.AdamW(
            G_params, lr=self.hparams.initial_learning_rate)

        D_sch = transformers.get_cosine_schedule_with_warmup(
            D_opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.max_steps // 2
        )
        G_sch = transformers.get_cosine_schedule_with_warmup(
            G_opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.max_steps // 2
        )

        return (
            [D_opt, G_opt],
            [{"scheduler": D_sch, "interval": "step"}, {
                "scheduler": G_sch, "interval": "step"}],
        )

    def forward(self, batch: dict):
        y_hat, commit_loss, vq_loss = self.G(
            duration_tokens=batch["duration_tokens"],
            phone=batch["phone_tokens"],
            phone_lens=batch["tokens_lens"],
            mel_mrte=batch["mel_timbres"],
            mel_vqpe=batch["mel_targets"]
        )

        return y_hat, commit_loss, vq_loss

    def training_step(self, batch: dict, batch_idx, **kwargs):
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        with torch.cuda.amp.autocast(dtype=self.train_dtype):
            self.G.train()
            y_hat, G_loss_commit, G_loss_vq = self(batch)

            # Train discriminator
            y = batch["mel_targets"]
            D_outputs = self.D(y)
            D_loss_real = 0.5 * torch.mean((D_outputs["y"] - 1) ** 2)

            D_outputs = self.D(y_hat.detach())
            D_loss_fake = 0.5 * torch.mean(D_outputs["y"] ** 2)

            D_loss_total = D_loss_real + D_loss_fake

            opt1.zero_grad()
            self.manual_backward(D_loss_total)
            opt1.step()
            sch1.step()

            # Train generator
            G_loss_re = F.l1_loss(y, y_hat)

            G_loss = G_loss_re + G_loss_commit * self.hparams.G_commit_loss_coeff + \
                G_loss_vq * self.hparams.G_vq_loss_coeff

            G_loss_adv = 0.5 * torch.mean((self.D(y_hat)["y"] - 1) ** 2)
            G_loss_total = G_loss_adv * self.hparams.G_adv_loss_coeff + G_loss

            opt2.zero_grad()
            self.manual_backward(G_loss_total)
            opt2.step()
            sch2.step()

        if batch_idx % 5 == 0:
            self.log("train/D_loss_total", D_loss_total, prog_bar=True)
            self.log("train/D_loss_real", D_loss_real)
            self.log("train/D_loss_fake", D_loss_fake)

            self.log("train/G_loss_total", G_loss_total, prog_bar=True)
            self.log("train/G_loss_adv", G_loss_adv)
            self.log("train/G_loss", G_loss)
            self.log("train/G_loss_commit", G_loss_commit)
            self.log("train/G_loss_vq", G_loss_vq)
            self.log("train/G_loss_re", G_loss_re)

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch: torch.Tensor, **kwargs):

        y = batch["mel_targets"]
        with torch.no_grad():
            self.G.eval()
            y_hat, _, _ = self(batch)

        loss_re = F.l1_loss(y, y_hat)

        self.validation_step_outputs.append({
            "y": y[0],
            "y_hat": y_hat[0],
            "loss_re": loss_re,
        })

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_rank == 0:

            mel = outputs[0]["y"].transpose(0, 1)
            mel_hat = outputs[0]["y_hat"].transpose(0, 1)

            self.logger.experiment.add_image(
                "val/mel_analyse",
                plot_spectrogram_to_numpy(
                    mel.data.cpu().numpy(), mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

            with torch.no_grad():
                hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz")
                hifi_gan.eval()

                audio_target = hifi_gan.decode_batch(mel.unsqueeze(0).cpu())
                audio_hat = hifi_gan.decode_batch(mel_hat.unsqueeze(0).cpu())

            self.logger.experiment.add_audio(
                "val/audio_target",
                audio_target[0],
                self.global_step,
                sample_rate=HIFIGAN_SR,
            )

            self.logger.experiment.add_audio(
                "val/audio_hat",
                audio_hat[0],
                self.global_step,
                sample_rate=HIFIGAN_SR,
            )

        loss_re = torch.mean(torch.stack(
            [x["loss_re"] for x in outputs]))

        self.log("val/loss_re", loss_re, sync_dist=True)

        self.validation_step_outputs = []
