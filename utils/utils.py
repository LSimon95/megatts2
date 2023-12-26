import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

import numpy as np


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

def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Save a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure object.

    Returns:
        ndarray: Numpy array representing the figure.
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spec_target: np.ndarray, spec_output: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.set_title("Target")
    im = ax1.imshow(spec_target.astype(np.float32), aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax1)
    plt.xlabel("Frames")
    plt.ylabel("Channels")

    ax2.set_title("Output")
    im = ax2.imshow(spec_output.astype(np.float32), aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax2)
    plt.xlabel("Frames")
    plt.ylabel("Channels")

    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data