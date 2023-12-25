import lightning.pytorch as pl

import torch
import torchaudio
import torch.nn.functional as F

import transformers

import numpy as np
import math

from .megatts2 import MegaVQ
from modules.dscrm import Discriminator

class MegaGANTrainer(pl.LightningModule):
    def __init__(
            self,
            model: MegaVQ,
            dscrm: Discriminator,
            g_lr: float,
            d_lr: float,
            warmup_steps: float = 45,

            gen_loss_coeff: float = 1.0,
            dscrm_loss_coeff: float = 1.0,

            train_dtype: str = "float32",
            **kwargs
            ):

        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['model', 'dscrm'])
        self.model = model

        self.dscrm = dscrm
        self.validation_step_outputs = []

        if self.hparams.train_dtype == "float32":
            self.train_dtype = torch.float32
        elif self.hparams.train_dtype == "bfloat16":
            self.train_dtype = torch.bfloat16
            print("Using bfloat16")

    def configure_optimizers(self):
        dscrm_params = [
            {"params": self.dscrm.parameters()}
        ]
        gen_params = [
            {"params": self.model.parameters()}
        ]

        opt_dscrm = torch.optim.AdamW(dscrm_params, lr=self.hparams.d_lr)
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.g_lr)

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_dscrm = transformers.get_cosine_schedule_with_warmup(
            opt_dscrm, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=max_steps
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=max_steps
        )

        return (
            [opt_dscrm, opt_gen],
            [{"scheduler": scheduler_dscrm, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def forward(self, batch : dict):
        y_hat, commit_loss, vq_loss = self.model(
            duration_tokens=batch["duration_tokens"],
            text=batch["phone_tokens"],
            text_lens=batch["tokens_lens"],
            mel_mrte=batch["mel_timbres"],
            mel_lens_mrte=batch["mel_timbre_lens"],
            mel_vqpe=batch["mel_targets"]
        )

        return y_hat, commit_loss, vq_loss

    def training_step(self, batch : dict, batch_idx, **kwargs):
        opt1, opt2 =  self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        # Train discriminator
        y = batch["mel_targets"]
        # dscrm_outputs = self.dscrm(y)
        # loss_dscrm_real = torch.mean((dscrm_outputs["y"] - 1) ** 2)

        # with torch.no_grad():
        #     self.model.eval()
        #     y_hat = self(batch)[0]
        # dscrm_outputs = self.dscrm(y_hat.detach())
        # loss_dscrm_fake = torch.mean(dscrm_outputs["y"] ** 2)

        # loss_dscrm = loss_dscrm_real + loss_dscrm_fake

        # opt1.zero_grad()
        # self.manual_backward(loss_dscrm)
        # opt1.step()
        # sch1.step()

        # Train generator
        with torch.cuda.amp.autocast(dtype=self.train_dtype):
            self.model.train()
            y_hat, commit_loss, vq_loss = self(batch)

            loss_gen_re = F.l1_loss(y, y_hat)
            
            loss_gen = loss_gen_re # + commit_loss + vq_loss

        opt2.zero_grad()
        self.manual_backward(loss_gen)
        opt2.step()
        sch2.step()

        if batch_idx % 5 == 0:

            # self.log("train/dscrm_loss_total", loss_dscrm, prog_bar=True)
            # self.log("train/dscrm_loss_real", loss_dscrm_real)
            # self.log("train/dscrm_loss_fake", loss_dscrm_fake)

            self.log("train/gen_loss_total", loss_gen, prog_bar=True)
            # self.log("train/gen_commit_loss", commit_loss)
            # self.log("train/gen_vq_loss", vq_loss)
            self.log("train/gen_loss_gen_re", loss_gen_re)

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch : torch.Tensor, **kwargs):
        pass

    def on_validation_epoch_end(self):
        pass

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx