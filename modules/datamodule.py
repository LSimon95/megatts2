import lightning.pytorch as pl

import random
from concurrent.futures import ThreadPoolExecutor

from lhotse import CutSet, load_manifest
from lhotse.dataset.collation import collate_features
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.dataset.input_strategies import (
    _get_executor
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, Tuple, List, Type

from utils.symbol_table import SymbolTable

from tqdm.auto import tqdm

import numpy as np

from modules.mrte import LengthRegulator

from .tokenizer import HIFIGAN_SR, HIFIGAN_HOP_LENGTH


class TokensCollector():
    def __init__(self, symbols_table: str) -> None:
        unique_tokens = SymbolTable.from_file(symbols_table).symbols
        self.token2idx = {token: idx for idx,
                          token in enumerate(unique_tokens)}

    def __call__(self, cuts: CutSet) -> (List, torch.Tensor):

        phone_tokens_list = []
        duration_tokens_list = []
        lens = []
        for cut in cuts:
            phone_tokens = cut.supervisions[0].custom['phone_tokens']
            duration_tokens = cut.supervisions[0].custom['duration_tokens']

            phone_tokens_list.append(self.phone2token(phone_tokens))
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

        phone_tokens = torch.stack(phone_tokens_list_padded)
        duration_tokens = torch.stack(
            duration_tokens_list_padded).type(torch.int64)
        lens = torch.Tensor(lens).to(dtype=torch.int32)

        return phone_tokens, duration_tokens, lens

    def phone2token(self, phone: List) -> int:
        return torch.Tensor(
            [self.token2idx[token] for token in phone]
        ).type(torch.int64)


class TTSDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            spk2cuts: Dict,
            ds_path: str,
            n_same_spk_samples: int = 10
    ):
        super().__init__()
        self.tokens_collector = TokensCollector(
            f'{ds_path}/unique_text_tokens.k2symbols')
        self.spk2cuts = spk2cuts
        self.n_same_spk_samples = n_same_spk_samples

    def __getitem__(self, cuts: CutSet) -> Dict:
        phone_tokens, duration_tokens, tokens_lens = self.tokens_collector(
            cuts)
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
            same_spk_cuts = same_spk_cuts.sample(
                n_cuts=min(n_sample, len(same_spk_cuts)))

            mel_timbres_same_spk, mel_timbre_lens_same_spk = collate_features(
                same_spk_cuts,
                executor=_get_executor(8, executor_type=ThreadPoolExecutor),)

            mel_timbre = mel_timbres_same_spk[0, :mel_timbre_lens_same_spk[0]]
            for i in range(1, mel_timbres_same_spk.shape[0]):
                mel_timbre = torch.cat(
                    [mel_timbre, mel_timbres_same_spk[i, :mel_timbre_lens_same_spk[i]]], dim=0)
            mel_timbres_list.append(mel_timbre)
            mel_timbre_lens_list.append(mel_timbre.shape[0])

        mel_timbres_list_cutted = []
        min_mel_timbres_len = min(mel_timbre_lens_list)
        for mel_timbre in mel_timbres_list:
            mel_timbres_list_cutted.append(mel_timbre[:min_mel_timbres_len, :])

        mel_timbres = torch.stack(mel_timbres_list_cutted).type(torch.float32)

        batch = {
            "phone_tokens":  phone_tokens,
            "duration_tokens": duration_tokens,
            "tokens_lens": tokens_lens,
            "mel_targets": mel_targets,
            "mel_target_lens": mel_target_lens,
            "mel_timbres": mel_timbres,
        }

        return batch


class MegaPLMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            spk2cuts: Dict,
            ds_path: str,
            lr: LengthRegulator,
            n_same_spk_samples: int = 10,
            vq_bins: int = 1024,

    ):
        super().__init__()
        self.spk2cuts = spk2cuts

        self.bos = torch.Tensor([vq_bins])
        self.eos = torch.Tensor([vq_bins + 1])

        self.n_same_spk_samples = n_same_spk_samples
        self.lr = lr

        self.tokens_collector = TokensCollector(
            f'{ds_path}/unique_text_tokens.k2symbols')
        self.ds_path = ds_path

    def read_latent(self, cut) -> Dict:

        id = cut.recording_id
        spk = cut.supervisions[0].speaker

        latents = np.load(f'{self.ds_path}/latents/{spk}/{id}.npy',
                          allow_pickle=True).item()
        tc_latent = torch.from_numpy(latents['tc_latent'])
        duration_tokens = torch.Tensor(
            cut.supervisions[0].custom['duration_tokens']).unsqueeze(0).to(dtype=torch.int32)
        tc_latent = self.lr(tc_latent, duration_tokens)
        p_code = torch.from_numpy(latents['p_code'])
        tc_latent = F.max_pool1d(tc_latent.transpose(
            1, 2), 8, ceil_mode=True).transpose(1, 2)

        return tc_latent, p_code

    def __getitem__(self, cuts_sample: CutSet) -> Dict:

        p_code_spks = []
        tc_latent_spks = []
        lens = []

        for cut in cuts_sample:

            spk = cut.supervisions[0].speaker

            same_spk_cuts = self.spk2cuts[spk]
            same_spk_cuts = same_spk_cuts.sample(
                n_cuts=self.n_same_spk_samples)

            tc_latent, p_code = self.read_latent(cut)

            tc_latent_spk = tc_latent[0, ...]
            p_code_spk = torch.cat([p_code[0, 0, :]])

            assert tc_latent_spk.shape[0] == p_code_spk.shape[0]

            for cut_spk in same_spk_cuts:
                tc_latent_spk_cat, p_code_spk_cat = self.read_latent(cut_spk)

                tc_latent_spk = torch.cat(
                    [tc_latent_spk_cat[0, ...], tc_latent_spk], dim=0)
                p_code_spk = torch.cat(
                    [p_code_spk_cat[0, 0, :], p_code_spk], dim=0)

                assert tc_latent_spk.shape[0] == p_code_spk.shape[0]
                assert torch.max(p_code_spk) < 1024

            p_code_spk = torch.cat([self.bos, p_code_spk], dim=0)
            lens.append(p_code_spk.shape[0] - 1)

            p_code_spks.append(p_code_spk)
            tc_latent_spks.append(tc_latent_spk)

        max_len = max(lens)

        # pad
        p_code_spks_padded = []
        tc_latent_spks_padded = []

        for i in range(len(p_code_spks)):
            p_code_spks_padded.append(F.pad(
                p_code_spks[i], (0, max_len - lens[i]), mode='constant', value=self.eos.item()))
            tc_latent_spks_padded.append(F.pad(
                tc_latent_spks[i], (0, 0, 0, max_len - lens[i]), mode='constant', value=0))

        p_code_spks = torch.stack(p_code_spks_padded).type(torch.int64)
        tc_latent_spks = torch.stack(tc_latent_spks_padded).type(torch.float32)
        lens = torch.Tensor(lens).to(dtype=torch.int32)

        batch = {
            "p_codes": p_code_spks,
            "tc_latents": tc_latent_spks,
            "lens": lens,
        }

        return batch


class MegaADMDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: str):
        self.tokens_collector = TokensCollector(
            f'{ds_path}/unique_text_tokens.k2symbols')
        self.ds_path = ds_path
        self.max_duration_token = 128

    def __getitem__(self, cuts_sample: CutSet) -> Dict:
        duration_token_list = []
        tc_latent_list = []
        lens = []
        for cut in cuts_sample:
            spk = cut.supervisions[0].speaker
            id = cut.recording_id

            duration_tokens = cut.supervisions[0].custom['duration_tokens']
            if np.max(duration_tokens) >= self.max_duration_token:
                continue

            duration_tokens = torch.Tensor(
                duration_tokens).to(dtype=torch.int32)

            latents = np.load(f'{self.ds_path}/latents/{spk}/{id}.npy',
                              allow_pickle=True).item()
            tc_latent = torch.from_numpy(latents['tc_latent'])[0]
            assert tc_latent.shape[0] == duration_tokens.shape[0]

            duration_token_list.append(duration_tokens)
            tc_latent_list.append(tc_latent)
            lens.append(duration_tokens.shape[0])

        max_len = max(lens)

        # pad
        duration_token_list_padded = []
        tc_latent_list_padded = []
        for i in range(len(duration_token_list)):
            duration_token_list_padded.append(F.pad(
                duration_token_list[i], (1, max_len - lens[i]), mode='constant', value=0))
            tc_latent_list_padded.append(F.pad(
                tc_latent_list[i], (0, 0, 0, max_len - lens[i]), mode='constant', value=0))

        duration_tokens = torch.stack(duration_token_list_padded).type(
            torch.float32).unsqueeze(-1)
        tc_latents = torch.stack(tc_latent_list_padded).type(torch.float32)
        lens = torch.Tensor(lens).to(dtype=torch.int32)

        batch = {
            "duration_tokens": duration_tokens,
            "tc_latents": tc_latents,
            "lens": lens,
        }

        return batch


def make_spk_cutset(cuts: CutSet) -> Dict[str, CutSet]:
    spk2cuts = {}
    for cut in tqdm(cuts, desc="Making spk2cuts"):
        spk = cut.supervisions[0].speaker
        if spk not in spk2cuts:
            spk2cuts[spk] = cuts.filter(
                lambda c: c.supervisions[0].speaker == spk).to_eager()

    return spk2cuts


class TTSDataModule(pl.LightningDataModule):
    def __init__(
            self,
            ds_path: str = 'data',
            max_duration_batch: float = 80,
            min_duration: float = 1.5,
            max_duration: float = 20,
            max_n_cuts: int = 3,
            num_buckets: int = 2,
            num_workers: int = 4,
            dataset: str = 'TTSDataset',
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

        if not self.hparams.dataset == 'MegaADMDataset':
            spk2cuts = make_spk_cutset(cs_train)

        if self.hparams.dataset == 'TTSDataset' or self.hparams.dataset == 'MegaADMDataset':
            if self.hparams.dataset == 'TTSDataset':
                dataset = TTSDataset(spk2cuts, self.hparams.ds_path, 10)
            else:
                dataset = MegaADMDataset(self.hparams.ds_path)

            sampler = DynamicBucketingSampler(
                cs_train,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=False,
                seed=seed,
            )
        elif self.hparams.dataset == 'MegaPLMDataset':
            lr = LengthRegulator(
                HIFIGAN_HOP_LENGTH, 16000, (HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000))
            dataset = MegaPLMDataset(
                spk2cuts, self.hparams.ds_path, lr, 10, 1024)

            sampler = SimpleCutSampler(
                cs_train,
                max_cuts=self.hparams.max_n_cuts,
                shuffle=True,
                drop_last=False,
                seed=seed,
            )
        else:
            raise ValueError(f'Unsupported dataset: {self.hparams.dataset}')

        self.train_dl = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

        cs_valid = load_manifest(f'{self.hparams.ds_path}/cuts_valid.jsonl.gz')
        cs_valid = cs_valid.filter(filter_duration)

        if not self.hparams.dataset == 'MegaADMDataset':
            spk2cuts = make_spk_cutset(cs_valid)

        if self.hparams.dataset == 'TTSDataset' or self.hparams.dataset == 'MegaADMDataset':
            sampler = DynamicBucketingSampler(
                cs_valid,
                max_duration=self.hparams.max_duration_batch,
                shuffle=True,
                num_buckets=self.hparams.num_buckets,
                drop_last=False,
                seed=seed,
            )
        elif self.hparams.dataset == 'MegaPLMDataset':
            sampler = SimpleCutSampler(
                cs_valid,
                max_cuts=self.hparams.max_n_cuts,
                shuffle=True,
                drop_last=False,
                seed=seed,
            )
        else:
            raise ValueError(f'Unsupported dataset: {self.hparams.dataset}')

        self.valid_dl = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_dl

    def val_dataloader(self) -> DataLoader:
        return self.valid_dl

    def test_dataloader(self) -> DataLoader:
        return None


def test():

    cs_valid = load_manifest("data/ds/cuts_valid.jsonl.gz")
    spk2cuts = make_spk_cutset(cs_valid)

    valid_dl = DataLoader(
        TTSDataset(spk2cuts, 'data/ds', 10),
        batch_size=None,
        num_workers=0,
        sampler=DynamicBucketingSampler(
            cs_valid,
            max_duration=10,
            shuffle=True,
            num_buckets=5,
            drop_last=False,
            seed=20000
        ),
    )

    # for batch in valid_dl:
    #     print(batch['phone_tokens'].shape)
    #     print(batch['duration_tokens'].shape)
    #     print(batch['tokens_lens'].shape)
    #     print(batch['mel_targets'].shape)
    #     print(batch['mel_target_lens'].shape)
    #     print(batch['mel_timbres'].shape)
    #     break

    lr = LengthRegulator(240, 16000, 15)

    valid_dl = DataLoader(
        MegaPLMDataset(spk2cuts, 'data/ds', lr, 10,
                       (HIFIGAN_HOP_LENGTH / HIFIGAN_SR * 1000)),
        batch_size=None,
        num_workers=0,
        sampler=SimpleCutSampler(
            cs_valid,
            max_cuts=3,
            shuffle=True,
            drop_last=True,
            seed=20000,
        ),
    )

    # for batch in valid_dl:
    #     print(batch['p_codes'].shape)
    #     print(batch['tc_latents'].shape)
    #     print(batch['lens'].shape)

    cs = cs_valid + load_manifest("data/ds/cuts_train.jsonl.gz")

    valid_dl = DataLoader(
        MegaADMDataset('data/ds'),
        batch_size=None,
        num_workers=0,
        sampler=DynamicBucketingSampler(
            cs,
            max_duration=20,
            shuffle=True,
            num_buckets=5,
            drop_last=False,
            seed=20000
        ),
    )

    for batch in valid_dl:
        pass
        # print(batch['duration_tokens'].shape)
        # print(batch['tc_latents'].shape)
        # print(batch['lens'].shape)
