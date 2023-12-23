import pytorch_lightning as pl

import torch
import torchaudio

import transformers

import numpy as np
import math

from .megatts2 import MegaGAN
from modules.dscrm import Discriminator


class MegaGANTrainer(pl.LightningModule):
    def __init__(
            self,
            model: MegaGAN,
            dscrm: Discriminator,
            g_lr: float,
            d_lr: int = 0,
            n_warmup_steps: float = 45,

            mel_loss_coeff: float = 1.0,
            disc_loss_coeff: float = 1.0,
            **kwargs
            ):

        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['model', 'dscrm'])
        self.model = model

        self.dscrm = dscrm
        self.validation_step_outputs = []


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
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_dscrm, num_warmup_steps=self.hparams.n_warmup_steps, num_training_steps=max_steps
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.n_warmup_steps, num_training_steps=max_steps
        )

        return (
            [opt_dscrm, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def forward(self, batch : dict):
        y_hat, commit_loss = self.model(
            duration_tokens=batch["duration_tokens"],
            text=batch["phone_tokens"],
            text_lens=batch["tokens_lens"],
            mel_mrte=batch["mel_targets"],
            mel_lens_mrte=batch["mel_target_lens"],
            mel_vqpe=batch["mel_timbres"]
        )

        return y_hat, commit_loss

    def training_step(self, batch : dict, **kwargs):
        opt1, opt2 =  self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        # Train discriminator
        y = batch["mel_targets"]
        dscrm_outputs = self.dscrm(y)
        loss_dscrm = self.dscrm_loss(dscrm_outputs)

        with torch.no_grad():
            self.model.eval()
            y_hat, commit_loss = self(batch)

            loss_dscr

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)

        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()
        sch1.step()

        # train generator
        self.codec.train()
        audio_hat, commit_loss = self(audio_input, **kwargs)
        if self.train_discriminator:
            _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                y=audio_input, y_hat=audio_hat, **kwargs,
            )
            loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
            loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
            loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

            self.log("generator/multi_period_loss", loss_gen_mp)
            self.log("generator/multi_res_loss", loss_gen_mrd)
            self.log("generator/feature_matching_mp", loss_fm_mp)
            self.log("generator/feature_matching_mrd", loss_fm_mrd)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        loss = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.hparams.mel_loss_coeff * mel_loss
            + self.hparams.commit_loss_coeff * commit_loss
        )

        self.log("generator/total_loss", loss, prog_bar=True)
        self.log("generator/mel_loss", mel_loss)
        self.log("generator/commit_loss", commit_loss)

        if self.global_step % 1000 == 0 and self.global_rank == 0:
            self.logger.experiment.add_audio(
                "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.codec.sample_rate
            )
            self.logger.experiment.add_audio(
                "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.codec.sample_rate
            )
            with torch.no_grad():
                mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
            self.logger.experiment.add_image(
                "train/mel_target",
                plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/mel_pred",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )

        opt2.zero_grad()
        self.manual_backward(loss)
        opt2.step()
        sch2.step()

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from .metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch : torch.Tensor, **kwargs):
        audio_input = batch
        self.codec.eval()
        audio_hat, _ = self(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.codec.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.codec.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from .metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        outputs = {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }
        self.validation_step_outputs.append(outputs)

        return outputs

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self.logger.experiment.add_audio(
                "val_in", audio_in.data.cpu().numpy(), self.global_step, self.codec.sample_rate
            )
            self.logger.experiment.add_audio(
                "val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.codec.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel/val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel/val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val/avg_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)