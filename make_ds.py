'''
    wavs dir
    ├── speaker1
    │   ├── s1wav1.wav
    │   ├── s1wav1.txt
    │   ├── s1wav2.wav
    │   ├── s1wav2.txt
    │   ├── ...
    ├── speaker2
    │   ├── s2wav1.wav
    │   ├── s2wav1.txt
    │   ├── ...

    to

    wavs dir
    ├── speaker1
    │   ├── s1wav1.wav
    │   ├── s1wav1.lab
    │   ├── s1wav2.wav
    │   ├── s1wav2.lab
    │   ├── ...
    ├── speaker2
    │   ├── s2wav1.wav
    │   ├── s2wav1.lab
    │   ├── ...
'''

import os

import glob
from modules.tokenizer import TextTokenizer
from multiprocessing import Pool
from tqdm.auto import tqdm

WAVTXT_PATH = '/workspace/megatts2data/wavlab'
N_T = 4

STAGE = 0

def make_lab(wav):
    name = wav.split('/')[-1].split('.')[0]
    folder = '/'.join(wav.split('/')[:-1])
    # Create lab files
    with open(f'{folder}/{name}.txt', 'r') as f:
        txt = f.read()

        with open(f'{folder}/{name}.lab', 'w') as f:
            f.write(' '.join(tt.tokenize(txt)))

    os.remove(f'{folder}/{name}.txt')

if __name__ == '__main__':

    # Create lab files
    if STAGE == 0:
        wavs = glob.glob(f'{WAVTXT_PATH}/**/*.wav', recursive=True)
        tt = TextTokenizer()

        with Pool(N_T) as p:
            for _ in tqdm(p.imap(make_lab, wavs), total=len(wavs)):
                pass
    
        
    # os.system(f'mfa align {WAVTXT_PATH}')
