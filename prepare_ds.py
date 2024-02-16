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

    cautions: stage 0 will delete all txt files in wavs dir
'''

import os

import glob
from modules.tokenizer import TextTokenizer
from multiprocessing import Pool
from tqdm.auto import tqdm
from utils.textgrid import read_textgrid

import argparse

from lhotse import validate_recordings_and_supervisions, CutSet, NumpyHdf5Writer, load_manifest_lazy, load_manifest
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import Seconds, compute_num_frames

from functools import partial

from modules.feat_extractor import (
    VOCODER_SR,
    VOCODER_HOP_SIZE,
    extract_mel_spec
)
from models.megatts2 import MegaG
from modules.datamodule import TTSDataset, make_spk_cutset

from utils.symbol_table import SymbolTable

import soundfile as sf
import librosa

import torch
import numpy as np

import h5py
import torchaudio as ta


def make_lab(tt, wav):
    id = wav.split('/')[-1].split('.')[0]
    folder = '/'.join(wav.split('/')[:-1])
    # Create lab files
    with open(f'{folder}/{id}.txt', 'r') as f:
        txt = f.read()

        tokens = tt.tokenize(txt)
        with open(f'{folder}/{id}.lab', 'w') as f:
            f.write(' '.join(tokens))

        return tokens


def extract_mel(args):
    spk, id = args
    wav = f'data/wavs/{spk}/{id}.wav'
    y, sr = ta.load(wav)
    mel_spec = extract_mel_spec(y)

    os.system(f'mkdir -p data/ds/mels/{spk}')
    torch.save(mel_spec, f'data/ds/mels/{spk}/{id}.pt')


class DatasetMaker:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--stage', type=int, default=0,
                            help='Stage to start from')
        parser.add_argument('--wavtxt_path', type=str,
                            default='data/wavs/', help='Path to wav and txt files')
        parser.add_argument('--text_grid_path', type=str,
                            default='data/textgrids/', help='Path to textgrid files')
        parser.add_argument('--ds_path', type=str,
                            default='data/ds/', help='Path to save dataset')
        parser.add_argument('--num_workers', type=int,
                            default=4, help='Number of workers')
        parser.add_argument('--test_set_ratio', type=float,
                            default=0.01, help='Test set ratio')
        parser.add_argument('--generator_ckpt', type=str,
                            default='generator.ckpt', help='Load generator checkpoint')
        parser.add_argument('--generator_config', type=str,
                            default='configs/config_gan.yaml', help='Load generator config')

        self.args = parser.parse_args()

        self.test_set_interval = int(1 / self.args.test_set_ratio)

    def make_labs(self):
        wavs = glob.glob(f'{self.args.wavtxt_path}/**/*.wav', recursive=True)
        tt = TextTokenizer()

        tokens_dict = set()
        for wav in tqdm(wavs):
            tokens_dict.update(make_lab(tt, wav))

        open(f'{self.args.ds_path}/tokens.txt', 'w').write(
            '\n'.join(sorted(list(tokens_dict))))
        # with Pool(self.args.num_workers) as p:
        #     for _ in tqdm(p.imap(partial(make_lab, tt), wavs), total=len(wavs)):
        #         pass

    def save_cuts(self, set_id, recordings, supervisions):
        set_parts = ['train', 'valid']
        for i in range(2):
            recording_set = RecordingSet.from_recordings(recordings[i])
            supervision_set = SupervisionSet.from_segments(supervisions[i])
            validate_recordings_and_supervisions(
                recording_set, supervision_set)

            supervision_set.to_file(
                f"{self.args.ds_path}/manifests/{set_id}_supervisions_{set_parts[i]}.jsonl.gz")
            recording_set.to_file(
                f"{self.args.ds_path}/manifests/{set_id}_recordings_{set_parts[i]}.jsonl.gz")

        manifests = read_manifests_if_cached(
            dataset_parts=['train', 'valid'],
            output_dir=f"{self.args.ds_path}/manifests/",
            prefix=f"{set_id}",
            suffix='jsonl.gz',
            types=["recordings", "supervisions"],
        )

        for partition, m in manifests.items():
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            cut_set.to_file(
                f"{self.args.ds_path}/manifests/{set_id}_cuts_{partition}.jsonl.gz")

    def make_ds(self):
        os.system(f'mkdir -p {self.args.ds_path}/manifests')
        tgs = glob.glob(
            f'{self.args.text_grid_path}/**/*.TextGrid', recursive=True)

        recordings = [[], []]  # train, test
        supervisions = [[], []]
        max_duration_token = 0
        spk_ids = []

        cs_cnt = 0
        for tg in tqdm(tgs):
            id = tg.split('/')[-1].split('.')[0]
            speaker = tg.split('/')[-2]

            spk_ids.append((speaker, id))

            intervals = [i for i in read_textgrid(tg) if (i[3] == 'phones')]

            y, sr = librosa.load(
                f'{self.args.wavtxt_path}/{speaker}/{id}.wav', sr=VOCODER_SR)

            frame_shift = VOCODER_HOP_SIZE / VOCODER_SR
            duration = round(y.shape[-1] / VOCODER_SR, ndigits=12)
            n_frames = compute_num_frames(
                duration=duration,
                frame_shift=frame_shift,
                sampling_rate=VOCODER_SR,
            )

            duration_tokens = []
            phone_tokens = []

            for i, interval in enumerate(intervals):
                n_frame_interval = int(interval[1] / frame_shift)
                duration_tokens.append(n_frame_interval - sum(duration_tokens))
                phone_tokens.append(interval[2] if interval[2] != '' else '_')

            if sum(duration_tokens) > n_frames:
                print(
                    f'{id} duration_tokens: {sum(duration_tokens)} must <= n_frames: {n_frames}')
                assert False

            recording = Recording.from_file(
                f'{self.args.wavtxt_path}/{speaker}/{id}.wav')
            text = open(
                f'{self.args.wavtxt_path}/{speaker}/{id}.txt', 'r').read()
            segment = SupervisionSegment(
                id=id,
                recording_id=id,
                start=0,
                duration=recording.duration,
                channel=0,
                language="CN",
                speaker=speaker,
                text=text,
            )

            set_id = 0 if i % self.test_set_interval else 1
            recordings[set_id].append(recording)
            supervisions[set_id].append(segment)

            segment.custom = {}
            segment.custom['duration_tokens'] = duration_tokens
            segment.custom['phone_tokens'] = phone_tokens

            max_duration_token = max(max_duration_token, len(duration_tokens))

            assert len(duration_tokens) == len(phone_tokens)

            if len(supervisions[0]) >= 65535:
                self.save_cuts(str(cs_cnt), recordings, supervisions)
                recordings = [[], []]
                supervisions = [[], []]
                cs_cnt += 1

        if len(supervisions[0]) > 0:
            self.save_cuts(str(cs_cnt), recordings, supervisions)

        # print(f'max_duration_token: {max_duration_token}')
        # # extract mel
        # with Pool(self.args.num_workers) as p:
        #     for _ in tqdm(p.imap(extract_mel, spk_ids), total=len(spk_ids), desc='Extracting mel'):
        #         pass

    def extract_latent(self):

        os.system(f'mkdir -p {self.args.ds_path}/latents')

        G = MegaG.from_pretrained(
            dm.args.generator_ckpt, dm.args.generator_config)
        G = G.cuda()
        G.eval()

        cs_all = load_manifest(f'{dm.args.ds_path}/cuts_train.jsonl.gz') + \
            load_manifest(f'{dm.args.ds_path}/cuts_valid.jsonl.gz')
        spk_cs = make_spk_cutset(cs_all)

        for spk in spk_cs.keys():
            os.system(f'mkdir -p {self.args.ds_path}/latents/{spk}')

        ttsds = TTSDataset(spk_cs, f'{dm.args.ds_path}', 10)

        for c in tqdm(cs_all):
            id = c.recording_id
            spk = c.supervisions[0].speaker
            batch = ttsds.__getitem__(CutSet.from_cuts([c]))

            s2_latent = {}
            with torch.no_grad():

                tc_latent, p_code = G.s2_latent(
                    batch['phone_tokens'].cuda(),
                    batch['tokens_lens'].cuda(),
                    batch['mel_timbres'].cuda(),
                    batch['mel_targets'].cuda()
                )

                s2_latent['tc_latent'] = tc_latent.cpu().numpy()
                s2_latent['p_code'] = p_code.cpu().numpy()

            np.save(f'{self.args.ds_path}/latents/{spk}/{id}.npy', s2_latent)


if __name__ == '__main__':
    dm = DatasetMaker()

    # Create lab files
    if dm.args.stage == 0:
        dm.make_labs()
    elif dm.args.stage == 1:
        dm.make_ds()

        # Test
        css = glob.glob(f'{dm.args.ds_path}/manifests/*_cuts_train.jsonl.gz')
        print('Train manifest:', len(css))

        cs_train = load_manifest(css[0])
        for i in range(1, len(css)):
            cs_train = cs_train + load_manifest(css[i])

        css = glob.glob(f'{dm.args.ds_path}/manifests/*_cuts_valid.jsonl.gz')
        print('Valid manifest:', len(css))

        cs_valid = load_manifest(css[0])
        for i in range(1, len(css)):
            cs_valid = cs_valid + load_manifest(css[i])
        cs = cs_train + cs_valid

        unique_symbols = set()

        for c in tqdm(cs):
            unique_symbols.update(c.supervisions[0].custom["phone_tokens"])

        unique_phonemes = SymbolTable()
        for s in sorted(list(unique_symbols)):
            unique_phonemes.add(s)

        unique_phonemes_file = f"unique_text_tokens.k2symbols"
        unique_phonemes.to_file(f'{dm.args.ds_path}/{unique_phonemes_file}')

        print(cs.describe())
    elif dm.args.stage == 2:
        dm.extract_latent()
