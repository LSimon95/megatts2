import torch

import numpy as np
import json

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames

from dataclasses import dataclass
from typing import Union

from bigvgan.meldataset import mel_spectrogram
from bigvgan.models import BigVGAN as Generator
from bigvgan.inference import load_checkpoint
from bigvgan.env import AttrDict

VOCODER_SR = 24000
VOCODER_HOP_SIZE = 256
VOCODER_WIN_SIZE = 1024
VOCODER_MEL_BINS = 100
VOCODER_NFFT = 1024
VOCODER_MAX_FREQ = 12000

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

def load_bigvgan_model(ckpt_dir):
    with open(f'{ckpt_dir}/config.json') as f:
        data = f.read()
    h = AttrDict(json.loads(data))

    generator = Generator(h).to('cpu')
    state_dict_g = load_checkpoint(f'{ckpt_dir}/g_05000000.zip', 'cpu')
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    return generator

def mel2wav(generator, mel):
    with torch.no_grad():
        return generator(mel)

# class MelSpecExtractor(FeatureExtractor):
#     name = "mel_spec"
#     config_type = AudioFeatExtraConfig

#     @property
#     def frame_shift(self) -> Seconds:
#         return self.config.frame_shift

#     def feature_dim(self, sampling_rate: int) -> int:
#         return self.config.feature_dim

#     def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray:
#         assert sampling_rate == VOCODER_SR
#         if not isinstance(samples, torch.Tensor):
#             samples = torch.from_numpy(samples)
#         torch.set_num_threads(1)
#         # Hifigan

#         samples = samples.squeeze()
#         mel_spec = extract_mel_spec(samples)

#         duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
#         num_frames = compute_num_frames(
#             duration=duration,
#             frame_shift=self.frame_shift,
#             sampling_rate=sampling_rate,
#         )
#         return mel_spec.squeeze(0).permute(1, 0)[:num_frames, :].numpy()

def test():
    import torchaudio as ta
    y, sr = ta.load('test.wav')
    mel_spec = extract_mel_spec(y)
    print(mel_spec.shape)

    # # test bigvgan
    # checkpoint_dict = load_checkpoint('/root/autodl-tmp/megatts2/bigvgan_base_24khz_100band/g_05000000.zip', 'cuda')


