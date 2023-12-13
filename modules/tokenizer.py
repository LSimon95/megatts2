from pypinyin import pinyin, Style, load_phrases_dict
from pypinyin.style._utils import get_finals, get_initials
from phonemizer.separator import Separator

from tn.chinese.normalizer import Normalizer

import re

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

if __name__ == '__main__':
    tt = TextTokenizer()

    txt = 'Hellow你好啊,我是Simon,你叫什么名字？What is your name?'
    phones = tt.phonemize(txt)
    print(phones)
    # assert phones  == 'hellow_ni3_hao3_wo3_shi4_simon_ni3_jiao4_shen2_me5_ming2_zi4_what_is_your_name'
    # print(tt.tokenize(txt))
    
