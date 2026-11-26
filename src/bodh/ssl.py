"""A small bidirectional spectrogram Transformer for masked patch reconstruction."""

from dataclasses import dataclass

import torch
from torch import nn

from bodh.audio import MEL_BANDS


@dataclass(frozen=True)
class EncoderConfig:
    patch_frames: int = 8
    max_patches: int = 32
    width: int = 128
    layers: int = 4
    heads: int = 4
    feedforward: int = 256
    dropout: float = 0.1

    @property
    def patch_size(self) -> int:
        return self.patch_frames * MEL_BANDS


class SpectrogramEncoder(nn.Module):
    """Encode full-band time patches with learned positions and bidirectional attention."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        # Build layers independently so each initializes from a separate RNG draw.
        self.projection = nn.Linear(config.patch_size, config.width)
        self.mask_token = nn.Parameter(torch.empty(config.width))
        self.position = nn.Parameter(torch.empty(config.max_patches, config.width))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                config.width, config.heads, config.feedforward, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.width)

    def forward(self, patches: torch.Tensor, valid: torch.Tensor,
                hidden: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.projection(patches.flatten(-2))
        if hidden is not None:
            tokens = torch.where(hidden.unsqueeze(-1), self.mask_token, tokens)
        tokens = tokens + self.position[:tokens.size(1)]
        for block in self.blocks:
            tokens = block(tokens, src_key_padding_mask=~valid)
        return self.norm(tokens)


class MaskedSpectrogramModel(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.encoder = SpectrogramEncoder(config)
        self.reconstruction = nn.Linear(config.width, config.patch_size)

    def forward(self, patches: torch.Tensor, valid: torch.Tensor,
                hidden: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(patches, valid, hidden)
        return self.reconstruction(encoded).reshape_as(patches)


def mask_patches(valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mask half the valid patches, rounded down with at least one reconstruction target."""
    draws = torch.rand(valid.shape, generator=generator).masked_fill(~valid, 2.0)
    ranks = draws.argsort(dim=1).argsort(dim=1)
    count = (valid.sum(dim=1) // 2).clamp_min(1)
    return valid & (ranks < count[:, None])
