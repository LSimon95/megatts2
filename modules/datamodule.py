import pytorch_lightning as pl

import glob
import random

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset.collation import collate_features
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.input_strategies import (
    _get_executor
)
from lhotse.utils import compute_num_frames
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, Tuple, List, Type

from utils.symbol_table import SymbolTable

from math import isclose

import numpy as np

class TokensCollector():
    def __init__(self, symbols_table: str, max_tokens: int) -> None:
        unique_tokens = SymbolTable.from_file(symbols_table).symbols
        assert len(unique_tokens) < max_tokens
        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}

    def __call__(self, cuts: CutSet) -> (List, torch.Tensor):

        phone_tokens_list = []
        duration_tokens_list = []
        lens = []
        for cut in cuts:
            phone_tokens = cut.supervisions[0].custom['phone_tokens']
            duration_tokens = cut.supervisions[0].custom['duration_tokens']

            phone_tokens_list.append(torch.Tensor(
                [self.token2idx[token] for token in phone_tokens]
            ))
            duration_tokens_list.append(torch.Tensor(duration_tokens))

            lens.append(len(phone_tokens))

        max_len = max(lens)
        phone_tokens_list_padded = []
        duration_tokens_list_padded = []
        for i in range(len(phone_tokens_list)):
            phone_tokens_list_padded.append(F.pad(
                phone_tokens_list[i], (0, max_len - lens[i]), mode='constant', value=0))
            duration_tokens_list_padded.append(F.pad(
                duration_tokens_list[i], (0, max_len - lens[i]), mode='constant', value=0))

        phone_tokens = torch.stack(phone_tokens_list_padded).type(torch.int64)
        duration_tokens = torch.stack(duration_tokens_list_padded).type(torch.int64)
        lens = torch.Tensor(lens).to(dtype=torch.int32)

        return phone_tokens, duration_tokens, lens


class TTSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            num_audio_tokens: int,
            symbols_table: str,
            train_stage: str = 'ar'
    ):
        super().__init__()

        self.text_tokens_collector = TextTokensCollector(
            symbols_table, num_audio_tokens)

        self.bos = self.text_tokens_collector.bos
        self.eos = num_audio_tokens
        self.train_stage = train_stage

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_tokens, text_token_lens = self.text_tokens_collector(
            cuts, self.train_stage == 'ar')

        audio_features = []
        audio_features_lens = []
        for cut in cuts:
            if cut.has_features:
                
                with h5py.File(cut.features.storage_path, "r") as f:
                    left_offset_frames, right_offset_frames = 0, 0
                    start = cut.start
                    duration = cut.duration

                    if not isclose(start, cut.features.start):
                        left_offset_frames = compute_num_frames(
                            start - cut.features.start,
                            frame_shift=cut.features.frame_shift,
                            sampling_rate=cut.features.sampling_rate,
                        )

                    right_offset_frames = left_offset_frames + compute_num_frames(
                        duration, frame_shift=cut.features.frame_shift, sampling_rate=cut.features.sampling_rate
                    )

                    audio_feature = f[cut.features.storage_key][left_offset_frames:right_offset_frames].copy()
                audio_features.append(audio_feature)
                audio_features_lens.append(len(audio_feature))

        audio_features_lens = torch.Tensor(audio_features_lens).to(dtype=torch.int32)

        # audio_features, audio_features_lens = collate_features(
        #     cuts,
        #     executor=_get_executor(8, executor_type=ThreadPoolExecutor),)

        # # print('audio f(before pad) ', audio_features[0, :audio_features_lens[0], 0], audio_features_lens[0])
        # audio_features = audio_features.transpose(1, 2)

        audio_features_list = []
        if self.train_stage == 'ar':
            max_audio_features_len = audio_features_lens.max().item() + 1 
            for i in range(len(audio_features)):
                audio_feature = torch.from_numpy(audio_features[i].astype(np.int64))[:, 0]
                audio_feature = F.pad(
                    audio_feature, (0, max_audio_features_len - audio_features_lens[i]), mode='constant', value=self.eos)
                audio_features_list.append(audio_feature)
            audio_features = torch.stack(audio_features_list)
            audio_features_lens += 1
        else:
            min_audio_features_len = audio_features_lens.min().item()
            for i in range(len(audio_features)):
                audio_feature = torch.from_numpy(audio_features[i].astype(np.int64)).transpose(0, 1)
                audio_features_list.append(audio_feature)
            audio_features = torch.stack(audio_features_list)
            audio_features_lens = torch.full_like(
                audio_features_lens, min_audio_features_len)

        # print('audio f(after pad) ', audio_features[0, :audio_features_lens[0]], audio_features_lens[0])

        batch = {
            "text_tokens":  text_tokens.to(torch.int64),
            "text_token_lens": text_token_lens,
            "audio_features": audio_features,
            "audio_feature_lens": audio_features_lens
        }

        return batch


class TTSDataModule(pl.LightningDataModule):
    def __init__(
            self,
            ds_path: str = 'data',
            max_duration_batch: float = 80,
            min_duration: float = 0,
            max_duration: float = 21,
            num_buckets: int = 2,
            symbols_table: str = 'data/unique_text_tokens.k2symbols',
            num_audio_tokens: int = 1024,
            num_workers: int = 4,
            train_stage: str = 'ar',
            **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None) -> None:

        def remove_short_and_long_utt(c):
            # Keep only utterances with duration between 0.6 second and 20 seconds
            if c.duration < self.hparams.min_duration or c.duration > self.hparams.max_duration:
                # logging.warning(
                #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
                # )
                return False
            return True

        seed = random.randint(0, 100000)

        cs_files = glob.glob(f'{self.hparams.ds_path}/*_cuts_train.jsonl.gz')
        print("Training Cuts: ", len(cs_files))
        random.shuffle(cs_files)
        cs_train = load_manifest_lazy(cs_files[0])
        for cs_file in cs_files[1:]:
            cs_train += load_manifest_lazy(cs_file)

        cs_train = cs_train.filter(remove_short_and_long_utt)

        self.train_dl = DataLoader(
            TTSDataset(
                num_audio_tokens=self.hparams.num_audio_tokens,
                symbols_table=self.hparams.symbols_table,
                train_stage=self.hparams.train_stage,
            ),
            batch_size=None,
            num_workers=self.hparams.num_workers,
            sampler=DynamicBucketingSampler(
                cs_train,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=True,
                # buffer_size=100000,
                # shuffle_buffer_size=200000,
                seed=seed
            ),
        )

        cs_files = glob.glob(f'{self.hparams.ds_path}/*_cuts_valid.jsonl.gz')
        print("Validation Cuts: ", len(cs_files))
        random.shuffle(cs_files)
        cs_valid = load_manifest_lazy(cs_files[0])
        for cs_file in cs_files[1:]:
            cs_valid += load_manifest_lazy(cs_file)

        cs_valid = cs_valid.filter(remove_short_and_long_utt)

        self.valid_dl = DataLoader(
            TTSDataset(
                num_audio_tokens=self.hparams.num_audio_tokens,
                symbols_table=self.hparams.symbols_table,
                train_stage=self.hparams.train_stage,
            ),
            batch_size=None,
            num_workers=self.hparams.num_workers,
            sampler=DynamicBucketingSampler(
                cs_valid,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=False,
                # buffer_size=100000,
                # shuffle_buffer_size=200000,
                seed=seed
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_dl

    def val_dataloader(self) -> DataLoader:
        return self.valid_dl

    def test_dataloader(self) -> DataLoader:
        return None


def test():

    cs = load_manifest_lazy(
        "/workspace/workpath/manifests/b_down_hq_0_cuts_train.jsonl.gz")

    dataloader = torch.utils.data.DataLoader(
        TTSDataset(
            1024, "/workspace/workpath/manifests/unique_text_tokens.k2symbols"),
        num_workers=1,
        batch_size=None,
        sampler=DynamicBucketingSampler(
            cs,
            max_duration=80,
            shuffle=True,
            num_buckets=32,
            drop_last=False,
            seed=43
        )
    )
    for batch in dataloader:
        print(batch['text_tokens'][0])
        print(batch['text_token_len'])
        print(batch['audio_features'][0])
        print(batch['audio_features_len'])
