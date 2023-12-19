import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths < lengths.unsqueeze(-1)


def make_attn_mask(lengths: torch.Tensor, num_heads: int) -> torch.Tensor:

    key_padding_mask = make_pad_mask(lengths)

    bsz = key_padding_mask.size(0)
    seq_len = key_padding_mask.size(1)

    return key_padding_mask.view(bsz, 1, 1, seq_len).expand(-1, num_heads, -1, -1)
