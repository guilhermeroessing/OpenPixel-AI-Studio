"""Testes de pos-processamento: pixelize, paleta, seamless e simetria."""
from __future__ import annotations

import numpy as np
import pytest

from openpixel_core import postprocess as pp
from openpixel_core.mock_provider import SmartMockProvider


def _hex_to_rgb(hexes):
    return {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes}


def _edge_mismatch(img: np.ndarray) -> float:
    """Diferença média entre bordas opostas (esquerda x direita e topo x base)."""
    f = img.astype(int)
    horiz = np.abs(f[:, 0] - f[:, -1]).mean()
    vert = np.abs(f[0] - f[-1]).mean()
    return float((horiz + vert) / 2)


# ---------------------------------------------------------------- tamanho de saida

@pytest.mark.parametrize("size", [16, 32, 64])
def test_pixelize_output_size_and_dtype(hr_image, size):
    out, palette = pp.pixelize(hr_image, size, palette_size=16)
    assert out.shape == (size, size, 3)
    assert out.dtype == np.uint8
    assert palette, "paleta nao pode ser vazia"


def test_downscale_fallback_produces_requested_size(hr_image):
    # pixeloe nao esta instalado no ambiente de teste -> cai no fallback INTER_AREA
    out = pp.downscale(hr_image, 32)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_downscale_upsamples_small_input_with_nearest(noise_image):
    small = noise_image[:8, :8]
    out = pp.downscale(small, 32)
    assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------- paleta

@pytest.mark.parametrize("palette_size", [2, 4, 8, 16])
def test_number_of_colors_respects_palette_size(hr_image, palette_size):
    out, palette = pp.pixelize(hr_image, 32, palette_size=palette_size)
    colors_in_image = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert len(colors_in_image) <= palette_size
    assert len(palette) <= palette_size
    assert len(set(palette)) == len(palette), "hex da paleta deve ser unico"


def test_palette_hex_matches_pixels(hr_image):
    out, palette = pp.pixelize(hr_image, 32, palette_size=8)
    colors_in_image = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert colors_in_image == _hex_to_rgb(palette)
    for h in palette:
        assert len(h) == 7 and h.startswith("#")


def test_kmeans_palette_returns_at_most_k_colors(hr_image):
    small = pp.downscale(hr_image, 32)
    pal = pp.kmeans_palette(small, 6)
    assert pal.dtype == np.uint8
    assert pal.shape[1] == 3
    assert 1 <= len(pal) <= 6


def test_fixed_palette_is_respected(hr_image):
    fixed = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], np.uint8)
    out, palette = pp.pixelize(hr_image, 32, palette=fixed)
    colors_in_image = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert colors_in_image <= {(0, 0, 0), (255, 255, 255), (255, 0, 0)}


def test_pixelize_is_deterministic(hr_image):
    a, pa = pp.pixelize(hr_image, 32, palette_size=16)
    b, pb = pp.pixelize(hr_image, 32, palette_size=16)
    assert np.array_equal(a, b)
    assert pa == pb


# ---------------------------------------------------------------- seamless

def test_make_seamless_reduces_edge_mismatch(noise_image):
    plain = _edge_mismatch(noise_image)
    seamless = _edge_mismatch(pp.make_seamless(noise_image))
    assert seamless < plain


def test_pixelize_seamless_reduces_edge_mismatch(noise_image):
    plain, _ = pp.pixelize(noise_image, 32, palette_size=16, seamless=False)
    seamless, _ = pp.pixelize(noise_image, 32, palette_size=16, seamless=True)
    assert _edge_mismatch(seamless) < _edge_mismatch(plain)


def test_pixelize_seamless_edges_are_close(noise_image):
    out, _ = pp.pixelize(noise_image, 32, palette_size=16, seamless=True)
    # bordas opostas devem estar na mesma vizinhanca de cor (ruido sem seamless fica ~15-17)
    assert _edge_mismatch(out) < 12


# ---------------------------------------------------------------- simetria

@pytest.mark.parametrize("seed", [1234, 1, 2, 3, 42])
def test_symmetric_mirrors_left_half_onto_right(seed):
    img, _ = SmartMockProvider().generate("tree forest", 256, seed)
    size = 32
    half = size // 2
    out, _ = pp.pixelize(img, size, palette_size=8, symmetric=True)
    right = out[:, size - half:]
    left_mirrored = out[:, :half][:, ::-1]
    assert np.array_equal(right, left_mirrored)


# ---------------------------------------------------------------- dither

def test_dither_keeps_colors_within_palette(hr_image):
    out, palette = pp.pixelize(hr_image, 32, palette_size=4, dither=True)
    colors_in_image = {tuple(c) for c in out.reshape(-1, 3).tolist()}
    assert colors_in_image <= _hex_to_rgb(palette)
    assert len(palette) <= 4


# ---------------------------------------------------------------- ilhas

def test_remove_islands_absorbs_single_pixel():
    idx = np.zeros((6, 6), np.int32)
    idx[3, 3] = 1  # ilha de 1 px
    out = pp.remove_islands(idx, np.zeros((2, 3), np.uint8), max_area=2)
    assert out[3, 3] == 0
    assert np.array_equal(out, np.zeros((6, 6), np.int32))
