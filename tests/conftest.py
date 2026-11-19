"""Configuração comum dos testes: torna importáveis os pacotes do monorepo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["DEVICE_MODE"] = "cpu_mock"  # testes sempre em mock, mesmo com .env em cuda

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "shared-contracts", ROOT / "backend-core", ROOT / "ai-gateway-mcp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture
def hr_image() -> np.ndarray:
    """Imagem de alta resolução determinística (mock do provider), 256x256 RGB uint8."""
    from openpixel_core.mock_provider import SmartMockProvider

    img, _ = SmartMockProvider().generate("grass tree rock", 256, 1234)
    return img


@pytest.fixture
def noise_image() -> np.ndarray:
    """Ruído uniforme 128x128 RGB, útil para checar bordas sem correlação."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
