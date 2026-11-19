"""Pixelizacao: grade fixa, paleta limitada/fixa e ampliacao sem suavizar."""
from __future__ import annotations

import numpy as np

from openpixel_core.pixelate import pixelate


def _noisy(size=256, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[:, : size // 2] = (200, 40, 40)
    base[:, size // 2:] = (30, 60, 220)
    return np.clip(base.astype(int) + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8)


def test_output_shape_and_dtype():
    out = pixelate(_noisy(), grid=32, colors=8, out_size=128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8


def test_grid_is_aligned_blocks():
    out = pixelate(_noisy(), grid=32, colors=8, out_size=128)  # cada pixel logico = 4x4
    blocks = out.reshape(32, 4, 32, 4, 3)
    assert (blocks == blocks[:, :1, :, :1, :]).all()


def test_palette_is_limited():
    out = pixelate(_noisy(), grid=32, colors=6, out_size=128)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 6


def test_fixed_palette_only_uses_given_colors():
    pal = [(255, 0, 0), (0, 0, 255), (0, 0, 0)]
    out = pixelate(_noisy(), grid=16, out_size=64, palette=pal, flat_bg=False)
    used = {tuple(int(v) for v in c) for c in np.unique(out.reshape(-1, 3), axis=0)}
    assert used <= set(pal)


def test_mode_method_runs():
    out = pixelate(_noisy(), grid=16, colors=8, out_size=64, method="mode")
    assert out.shape == (64, 64, 3)
