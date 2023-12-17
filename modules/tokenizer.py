from pypinyin import pinyin, Style, load_phrases_dict
from pypinyin.style._utils import get_finals, get_initials
from phonemizer.separator import Separator

from tn.chinese.normalizer import Normalizer

import re
from dataclasses import dataclass

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames

import numpy as np
import torch

from typing import Union

import torchaudio as ta

class TextTokenizer:
    def __init__(self) -> None:

        self.normalizer = Normalizer()
        self.separator = Separator(word="_", syllable="-", phone="|")

    def phonemize(self, text: str) -> str:
        text = self.normalizer.normalize(text)
        text = re.sub(r'<oov>.*</oov>', ' ', text)  # remove oov
        text = re.sub(r'[ ]+', ' ', text)  # remove extra spaces
        text = text.lower()

        phonemizeds = []
        for text_eng_chn in re.split(r"[^\w\s']+", text):
            # split chinese and english
            for text in re.split(r"([a-z ]+)", text_eng_chn):
                text = text.strip()
                if text == '' or text == "'":
                    continue
                    d = []
                if re.match(r"[a-z ']+", text):
                    for word in re.split(r"[ ]+", text):
                        phonemizeds.append(word)
                else:
                    phones = []
                    for n, py in enumerate(
                        pinyin(
                            text, style=Style.TONE3, neutral_tone_with_five=True
                        )
                    ):
                        if not py[0][-1].isalnum():
                            raise ValueError
                        phones.append(py[0])
                    phonemizeds.append(' '.join(phones))

        phonemizeds = f'{self.separator.word}'.join(
            [phones for phones in phonemizeds])
        return phonemizeds

    def tokenize(self, text):
        tokens = []
        for word in re.split('([_-])', self.phonemize(text.strip())):
            if len(word):
                for phone in re.split('\|', word):
                    if len(phone):
                        tokens.append(phone)
        return tokens

@dataclass
class AudioFeatExtraConfig:
    frame_shift: Seconds = 320.0 / 16000
    feature_dim: int = 128

class MelSpecExtractor(FeatureExtractor):
    name = "mel_spec"
    config_type = AudioFeatExtraConfig

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.feature_dim

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        torch.set_num_threads(1)

        mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=1024,
            win_length=480,
            hop_length=240,
            n_mels=128,
            f_min=0,
            f_max=None,
            power=1.0,
        )(samples)
        duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
        num_frames = compute_num_frames(
            duration=duration,
            frame_shift=self.frame_shift,
            sampling_rate=sampling_rate,
        )
        return mel_spec.squeeze(0).permute(1, 0)[:num_frames, :].numpy()


if __name__ == '__main__':
    tt = TextTokenizer()

    txt = 'Hellow你好啊,我是Simon,你叫什么名字？What is your name?'
    phones = tt.phonemize(txt)
    print(phones)
    # assert phones  == 'hellow_ni3_hao3_wo3_shi4_simon_ni3_jiao4_shen2_me5_ming2_zi4_what_is_your_name'
    # print(tt.tokenize(txt))
    
