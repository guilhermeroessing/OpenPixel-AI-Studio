"""service._base usa o pixelizador K-Means por padrao e o antigo com PIXELIZER=pixeloe."""
from __future__ import annotations

import numpy as np

from openpixel_core import service
from schemas import GenerateRequest


def _req(**kw):
    return GenerateRequest(prompt="knight", size=32, palette_size=8, seed=1, **kw)


def test_default_pixelizer_limits_palette_and_size():
    rgb, pal, _ = service._base(_req())
    assert rgb.shape == (32, 32, 3) and len(pal) <= 8 and len(pal) == len(np.unique(rgb.reshape(-1, 3), axis=0))


def test_pixeloe_switch(monkeypatch):
    monkeypatch.setenv("PIXELIZER", "pixeloe")
    rgb, pal, _ = service._base(_req())
    assert rgb.shape == (32, 32, 3) and len(pal) <= 8
