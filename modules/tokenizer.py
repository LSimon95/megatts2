from pypinyin import pinyin, Style
from phonemizer.separator import Separator
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from pypinyin.style._utils import get_finals, get_initials
import re

class TextTokenizer:
    def __init__(self) -> None:

        self.separator = Separator(word="_", syllable="-", phone="|")
        self.phonemizer = EspeakBackend(
            "en-us",
            punctuation_marks=Punctuation.default_marks(),
            preserve_punctuation=True,
            with_stress=False,
            tie=False,
            language_switch="keep-flags",
            words_mismatch="ignore",
        )
        print('TextTokenizer initialized')

    def phonemize(self, text: str) -> str:
        text = re.sub(r'[^\w\s]+', ' ', text)  # remove punctuation
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
                    phones = self.phonemizer.phonemize(
                        [text], separator=self.separator, strip=True, njobs=1)
                    # ['w|ɛ|ɹ|æ|f|_ɛ|s|d|iː|ɛ|f|dʒ|eɪ|ɛ|s|d|iː|dʒ|eɪ|ɛ|f|ɛ|s|d|iː|_ɛ|f|_dʒ|eɪ|d|iː|ɛ|f|dʒ|eɪ|s|æ|f|_ɛ|s|d|iː|ɛ|f|ɛ|s|dʒ|eɪ|d|iː|ɛ|f|_ɛ|s|d|iː|ɛ|f|_']
                    phones = phones[0].rstrip('_|').replace(
                        '_', self.separator.syllable)

                    phones_en_tags = []
                    for phone in re.split(f'([_|-])', phones):
                        if not phone == '' and not phone in '_|-':
                            phone = 'ipa' + phone
                        phones_en_tags.append(phone)
                    phonemizeds.append(''.join(phones_en_tags))
                else:
                    phones = []
                    for n, py in enumerate(
                        pinyin(
                            text, style=Style.TONE3, neutral_tone_with_five=True
                        )
                    ):
                        if not py[0][-1].isalnum():
                            raise ValueError

                        initial = get_initials(py[0], strict=False)
                        if py[0][-1].isdigit():
                            final = (
                                get_finals(py[0][:-1], strict=False)
                                + py[0][-1]
                            )
                        else:
                            final = get_finals(py[0], strict=False)
                        if initial:
                            phones.extend([initial])
                        phones.extend([final])

                    phonemizeds.append(self.separator.phone.join(phones))

        phonemizeds = f'{self.separator.word}'.join(
            [phones for phones in phonemizeds])
        return phonemizeds

    def tokenize(self, text):
        phones = []
        for word in re.split('(_)', self.phonemize(text.strip())):
            if len(word):
                for phone in re.split('[-\|]', word):
                    if len(phone):
                        phones.append(phone)

        return phones


if __name__ == '__main__':
    tt = TextTokenizer()

    txt = 'Hellow你好啊,我是Simon,你叫什么名字？What is your name?'
    phones = tt.phonemize(txt)
    print(phones)
    print(tt.tokenize(txt))

    # print(tt.tokenize_lty(tt.tokenize(txt)))
    # assert phones  == 'hellow_ni3_hao3_wo3_shi4_simon_ni3_jiao4_shen2_me5_ming2_zi4_what_is_your_name'
    # print(tt.tokenize(txt))
