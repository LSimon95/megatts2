from pypinyin import pinyin, Style
from phonemizer.separator import Separator

import re
from dataclasses import dataclass

from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames

import numpy as np
import torch

from typing import Union

import torchaudio as ta
import re

HIFIGAN_SR = 16000
HIFIGAN_HOP_LENGTH = 256
HIFIGAN_WIN_LENGTH = 1024
HIFIGAN_MEL_CHANNELS = 80
HIFIGAN_NFFT = 1024
HIFIGAN_MAX_FREQ = 8000


def get_pinyin2lty():
    pinyin2lty = {}
    with open('utils/mandarin_pinyin_to_mfa_lty.dict', 'r') as f:
        lines = f.readlines()

        for line in lines:
            ele = re.split(r'\t', line)

            ity_phones = re.split(r'[ ]+', ele[-1].strip())
            pinyin2lty[ele[0]] = ity_phones

    return pinyin2lty


class TextTokenizer:
    def __init__(self) -> None:

        self.separator = Separator(word="_", syllable="-", phone="|")
        self.pinyin2lty = get_pinyin2lty()

    def phonemize(self, text: str) -> str:
        text = re.sub(r'[^\w\s]+', ' ', text) # remove punctuation
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
                    phonemizeds.append(self.separator.phone.join(phones))

        phonemizeds = f'{self.separator.word}'.join(
            [phones for phones in phonemizeds])
        return phonemizeds

    def tokenize(self, text):
        phones = []
        for word in re.split('([_-])', self.phonemize(text.strip())):
            if len(word):
                for phone in re.split('\|', word):
                    if len(phone):
                        phones.append(phone)

        return phones

    def tokenize_lty(self, pinyin_tokens):
        lty_tokens_list = []

        for token in pinyin_tokens:
            if token in self.pinyin2lty.keys():
                lty_tokens = self.pinyin2lty[token]
                phones = self.separator.syllable.join(lty_tokens)
                lty_tokens = re.split(rf'({self.separator.syllable})', phones)
                lty_tokens_list += lty_tokens
            else:
                lty_tokens_list.append(token)
        return lty_tokens_list


@dataclass
class AudioFeatExtraConfig:
    frame_shift: Seconds = HIFIGAN_HOP_LENGTH / HIFIGAN_SR
    feature_dim: int = HIFIGAN_MEL_CHANNELS


class MelSpecExtractor(FeatureExtractor):
    name = "mel_spec"
    config_type = AudioFeatExtraConfig

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.feature_dim

    def extract(self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> np.ndarray:
        assert sampling_rate == HIFIGAN_SR
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        torch.set_num_threads(1)
        # Hifigan
        mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=HIFIGAN_NFFT,
            win_length=HIFIGAN_WIN_LENGTH,
            hop_length=HIFIGAN_HOP_LENGTH,
            n_mels=self.config.feature_dim,
            f_min=0,
            f_max=HIFIGAN_MAX_FREQ,
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
    print(tt.tokenize(txt))
    print(tt.tokenize_lty(tt.tokenize(txt)))
    # assert phones  == 'hellow_ni3_hao3_wo3_shi4_simon_ni3_jiao4_shen2_me5_ming2_zi4_what_is_your_name'
    # print(tt.tokenize(txt))
