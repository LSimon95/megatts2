import lightning.pytorch as pl

import torch
import torchaudio
import torch.nn.functional as F

import transformers

import numpy as np
import math

from .megatts2 import MegaG, MegaPLM, MegaADM
from modules.dscrm import Discriminator
from modules.feat_extractor import VOCODER_SR

from utils.utils import plot_spectrogram_to_numpy

from torchmetrics.classification import MulticlassAccuracy

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

        loss_re = torch.mean(torch.stack(
            [x["loss_re"] for x in outputs]))

        self.log("val/loss_re", loss_re, sync_dist=True)

        self.validation_step_outputs = []

class MegaPLMTrainer(pl.LightningModule):
    def __init__(
            self,
            plm: MegaPLM,
            initial_learning_rate: float,
            warmup_steps: float = 200,
            train_dtype: str = "float32",
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['plm'])
        self.validation_step_outputs = []

        if self.hparams.train_dtype == "float32":
            self.train_dtype = torch.float32
        elif self.hparams.train_dtype == "bfloat16":
            self.train_dtype = torch.bfloat16
            print("Using bfloat16")

        self.plm = plm

        self.accuracy_metric = MulticlassAccuracy(
            1024,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=1024 + 1
        )

    def configure_optimizers(self):
        plm_params = [
            {"params": self.plm.parameters()}
        ]

        plm_opt = torch.optim.AdamW(
            plm_params, lr=self.hparams.initial_learning_rate)

        plm_sch = transformers.get_cosine_schedule_with_warmup(
            plm_opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.max_steps
        )

        return (
            [plm_opt],
            [{"scheduler": plm_sch, "interval": "step"}],
        )
    
    def forward(self, batch: dict):
        logits, y = self.plm(
            tc_latent=batch["tc_latents"],
            p_codes=batch["p_codes"],
            lens=batch["lens"]
        )

        logits = logits.transpose(1, 2)

        # ignore padding
        loss = F.cross_entropy(logits, y, reduction="sum", ignore_index=1024 + 1)
        loss_log = loss / y.shape[0] / y.shape[1]
        ac10 = self.accuracy_metric(logits.detach(), y)

        return loss, loss_log, ac10
    
    def training_step(self, batch: dict, batch_idx, **kwargs):
        with torch.cuda.amp.autocast(dtype=self.train_dtype):
            self.plm.train()
            loss, loss_log, ac10 = self(batch)

        if batch_idx % 5 == 0:
            self.log("train/ac10", ac10, prog_bar=True)
            self.log("train/loss", loss_log, prog_bar=True)

        return loss
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch: torch.Tensor, **kwargs):
        with torch.no_grad():
            self.plm.eval()
            _, loss_log, ac10 = self(batch)

        self.validation_step_outputs.append({
            "loss_log": loss_log,
            "ac10": ac10
        })

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_rank == 0:
            loss_log = torch.mean(torch.stack(
                [x["loss_log"] for x in outputs]))
            ac10 = torch.mean(torch.stack(
                [x["ac10"] for x in outputs]))

            self.log("val/loss", loss_log, sync_dist=True)
            self.log("val/ac10", ac10, sync_dist=True)

        self.validation_step_outputs = []

class MegaADMTrainer(pl.LightningModule):
    def __init__(
            self,
            adm: MegaADM,
            initial_learning_rate: float,
            warmup_steps: float = 200,
            train_dtype: str = "float32",
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['adm'])
        self.validation_step_outputs = []

        if self.hparams.train_dtype == "float32":
            self.train_dtype = torch.float32
        elif self.hparams.train_dtype == "bfloat16":
            self.train_dtype = torch.bfloat16
            print("Using bfloat16")

        self.adm = adm

    def configure_optimizers(self):
        adm_params = [
            {"params": self.adm.parameters()}
        ]

        adm_opt = torch.optim.AdamW(
            adm_params, lr=self.hparams.initial_learning_rate)

        adm_sch = transformers.get_cosine_schedule_with_warmup(
            adm_opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.max_steps
        )

        return (
            [adm_opt],
            [{"scheduler": adm_sch, "interval": "step"}],
        )
    
    def forward(self, batch: dict):
        duration_tokens_predict, target = self.adm(
            tc_latents=batch["tc_latents"],
            duration_tokens=batch["duration_tokens"],
            lens=batch["lens"]
        )

        # ignore padding
        loss = F.mse_loss(duration_tokens_predict, target, reduction="sum")
        loss_log = loss / target.shape[0] / target.shape[1]

        return loss, loss_log
    
    def training_step(self, batch: dict, batch_idx, **kwargs):
        with torch.cuda.amp.autocast(dtype=self.train_dtype):
            self.adm.train()
            loss, loss_log = self(batch)

        if batch_idx % 5 == 0:
            self.log("train/loss", loss_log, prog_bar=True)

        return loss
    
    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch: torch.Tensor, **kwargs):
        with torch.no_grad():
            self.adm.eval()
            _, loss_log = self(batch)

        self.validation_step_outputs.append({
            "loss_log": loss_log
        })

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_rank == 0:
            loss_log = torch.mean(torch.stack(
                [x["loss_log"] for x in outputs]))

            self.log("val/loss", loss_log, sync_dist=True)

        self.validation_step_outputs = []