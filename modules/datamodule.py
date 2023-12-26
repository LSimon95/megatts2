import lightning.pytorch as pl

import glob
import random

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from lhotse import CutSet, load_manifest
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
    def __init__(self, symbols_table: str) -> None:
        unique_tokens = SymbolTable.from_file(symbols_table).symbols
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

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

class TTSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            spk2cuts: CutSet,
            symbols_table: str,
            n_same_spk_samples: int = 10
    ):
        super().__init__()
        self.tokens_collector = TokensCollector(symbols_table)
        self.spk2cuts = spk2cuts
        self.n_same_spk_samples = n_same_spk_samples

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phone_tokens, duration_tokens, tokens_lens = self.tokens_collector(cuts)
        mel_targets, mel_target_lens = collate_features(
            cuts,
            executor=_get_executor(8, executor_type=ThreadPoolExecutor),)
        
        # align duration token and mel_target_lens
        for i in range(mel_target_lens.shape[0]):
            sum_duration = torch.sum(duration_tokens[i])
            assert sum_duration <= mel_target_lens[i]
            if sum_duration < mel_target_lens[i]:
                mel_target_lens[i] = sum_duration
        
        max_len = max(mel_target_lens)
        mel_targets = mel_targets[:, :max_len, :]
        
        mel_timbres_list = []
        mel_timbre_lens_list = []
        n_sample = random.randint(2, self.n_same_spk_samples)
        for cut in cuts:
            same_spk_cuts = self.spk2cuts[cut.supervisions[0].speaker]
            same_spk_cuts = same_spk_cuts.sample(n_cuts=min(n_sample, len(same_spk_cuts)))

            mel_timbres_same_spk, mel_timbre_lens_same_spk = collate_features(
                same_spk_cuts,
                executor=_get_executor(8, executor_type=ThreadPoolExecutor),)

            mel_timbre = mel_timbres_same_spk[0, :mel_timbre_lens_same_spk[0]]
            for i in range(1, mel_timbres_same_spk.shape[0]):
                mel_timbre = torch.cat([mel_timbre, mel_timbres_same_spk[i, :mel_timbre_lens_same_spk[i]]], dim=0)
            mel_timbres_list.append(mel_timbre)
            mel_timbre_lens_list.append(mel_timbre.shape[0])
            

        max_len = max(mel_timbre_lens_list)
        mel_timbres_list_padded = []
        for i in range(len(mel_timbres_list)):
            mel_timbres_list_padded.append(F.pad(
                mel_timbres_list[i], (0, 0, 0, max_len - mel_timbre_lens_list[i]), mode='constant', value=0))

        mel_timbres = torch.stack(mel_timbres_list_padded).type(torch.float32)
        mel_timbre_lens = torch.Tensor(mel_timbre_lens_list).to(dtype=torch.int32)

        mel_targets = spectral_normalize_torch(mel_targets)
        mel_timbres = spectral_normalize_torch(mel_timbres)

        batch = {
            "phone_tokens":  phone_tokens,
            "duration_tokens": duration_tokens,
            "tokens_lens": tokens_lens,
            "mel_targets": mel_targets,
            "mel_target_lens": mel_target_lens,
            "mel_timbres": mel_timbres,
            "mel_timbre_lens": mel_timbre_lens
        }

        return batch

def make_spk_cutset(cuts: CutSet) -> Dict[str, CutSet]:
    spk2cuts = {}
    for cut in cuts:
        spk = cut.supervisions[0].speaker
        if spk not in spk2cuts:
            spk2cuts[spk] = cuts.filter(lambda c: c.supervisions[0].speaker == spk)

    return spk2cuts

class TTSDataModule(pl.LightningDataModule):
    def __init__(
            self,
            ds_path: str = 'data',
            max_duration_batch: float = 80,
            min_duration: float = 1.5,
            max_duration: float = 20,
            num_buckets: int = 2,
            num_workers: int = 4,
            **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['class_path'])

    def setup(self, stage: str = None) -> None:

        def filter_duration(c):
            if c.duration < self.hparams.min_duration or c.duration > self.hparams.max_duration:
                return False
            return True

        seed = random.randint(0, 100000)
        cs_train = load_manifest(f'{self.hparams.ds_path}/cuts_train.jsonl.gz')
        cs_train = cs_train.filter(filter_duration)

        spk2cuts = make_spk_cutset(cs_train)

        self.train_dl = DataLoader(
            TTSDataset(spk2cuts, f'{self.hparams.ds_path}/unique_text_tokens.k2symbols', 10),
            batch_size=None,
            num_workers=self.hparams.num_workers,
            sampler=DynamicBucketingSampler(
                cs_train,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=True,
                seed=seed
            ),
        )

        cs_valid = load_manifest(f'{self.hparams.ds_path}/cuts_valid.jsonl.gz')
        cs_valid = cs_valid.filter(filter_duration)

        spk2cuts = make_spk_cutset(cs_valid)

        self.valid_dl = DataLoader(
            TTSDataset(spk2cuts, f'{self.hparams.ds_path}/unique_text_tokens.k2symbols', 10),
            batch_size=None,
            num_workers=self.hparams.num_workers,
            sampler=DynamicBucketingSampler(
                cs_valid,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=False,
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

    cs = load_manifest("data/ds/cuts_train.jsonl.gz")

    dataloader = torch.utils.data.DataLoader(
        TTSDataset(cs, "data/ds/unique_text_tokens.k2symbols", 10),
        num_workers=1,
        batch_size=None,
        sampler=DynamicBucketingSampler(
            cs,
            max_duration=30,
            shuffle=True,
            num_buckets=32,
            drop_last=False,
            seed=43
        )
    )
    for batch in dataloader:
        print(batch["phone_tokens"].shape)
        print(batch["duration_tokens"].shape)
        print(batch["tokens_lens"].shape)
        print(batch["mel_targets"].shape)
        print(batch["mel_target_lens"].shape)
        print(batch["mel_timbres"].shape)
        print(batch["mel_timbre_lens"].shape)
