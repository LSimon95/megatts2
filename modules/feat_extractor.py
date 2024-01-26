import torch

import numpy as np

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames

from dataclasses import dataclass
from typing import Union

from bigvgan.meldataset import mel_spectrogram


VOCODER_SR = 16000
VOCODER_HOP_SIZE = 256
VOCODER_WIN_SIZE = 1024
VOCODER_MEL_BINS = 80
VOCODER_NFFT = 1024
VOCODER_MAX_FREQ = 8000


@dataclass
class AudioFeatExtraConfig:
    frame_shift: Seconds = VOCODER_HOP_SIZE / VOCODER_SR
    feature_dim: int = VOCODER_MEL_BINS


def extract_mel_spec(samples):
    return mel_spectrogram(
        y=samples,
        n_fft=VOCODER_NFFT,
        num_mels=VOCODER_MEL_BINS,
        sampling_rate=VOCODER_SR,
        hop_size=VOCODER_HOP_SIZE,
        win_size=VOCODER_WIN_SIZE,
        fmin=0,
        fmax=VOCODER_MAX_FREQ,
        center=False
    )


class MelSpecExtractor(FeatureExtractor):
    name = "mel_spec"
    config_type = AudioFeatExtraConfig

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.feature_dim

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray:
        assert sampling_rate == VOCODER_SR
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        torch.set_num_threads(1)
        # Hifigan

        samples = samples.squeeze()
        mel_spec = extract_mel_spec(samples)

        duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
        num_frames = compute_num_frames(
            duration=duration,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
        )
        return mel_spec.squeeze(0).permute(1, 0)[:num_frames, :].numpy()
