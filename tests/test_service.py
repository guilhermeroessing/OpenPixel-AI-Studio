"""Testes do service: sprite, 8 direcoes, animacao e paleta travada."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from openpixel_core import service
from schemas import GenerateRequest

EXPECTED_DIRECTIONS = {"S", "SE", "E", "NE", "N", "NW", "W", "SW"}


def _decode(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _hex_set(hexes):
    return {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes}


def _colors(img: np.ndarray) -> set:
    return {tuple(c) for c in img.reshape(-1, 3).tolist()}


@pytest.fixture
def req() -> GenerateRequest:
    return GenerateRequest(prompt="grass tree", size=32, palette_size=16, seed=42)


# ---------------------------------------------------------------- generate_pixel_art

def test_generate_pixel_art_returns_sprite(req):
    sprite = service.generate_pixel_art(req)
    assert sprite.width == 32 and sprite.height == 32
    assert sprite.seed == 42
    assert sprite.provider == "smart_mock"
    img = _decode(sprite.image_b64)
    assert img.shape == (32, 32, 3)


def test_generate_pixel_art_palette_bounds(req):
    sprite = service.generate_pixel_art(req)
    img = _decode(sprite.image_b64)
    assert len(sprite.palette) <= req.palette_size
    assert _colors(img) <= _hex_set(sprite.palette)


def test_generate_pixel_art_is_deterministic(req):
    a = service.generate_pixel_art(req)
    b = service.generate_pixel_art(req)
    assert a.image_b64 == b.image_b64
    assert a.palette == b.palette


# ---------------------------------------------------------------- 8 direcoes

def test_generate_8_directions_has_all_directions(req):
    sprites = service.generate_8_directions(req)
    assert set(sprites) == EXPECTED_DIRECTIONS
    for name, sprite in sprites.items():
        assert sprite.width == 32 and sprite.height == 32, name


def test_generate_8_directions_share_identical_palette(req):
    sprites = service.generate_8_directions(req)
    palettes = {tuple(s.palette) for s in sprites.values()}
    assert len(palettes) == 1, "todas as direcoes devem usar a mesma paleta"
    palette = next(iter(sprites.values())).palette
    for name, sprite in sprites.items():
        assert _colors(_decode(sprite.image_b64)) <= _hex_set(palette), name


def test_generate_8_directions_west_is_mirror_of_east(req):
    sprites = service.generate_8_directions(req)
    east = _decode(sprites["E"].image_b64)
    west = _decode(sprites["W"].image_b64)
    assert np.array_equal(west, east[:, ::-1])


def test_generate_8_directions_diagonals_differ_from_cardinals(req):
    sprites = service.generate_8_directions(req)
    imgs = {d: _decode(s.image_b64) for d, s in sprites.items()}
    assert not np.array_equal(imgs["SE"], imgs["E"])
    assert not np.array_equal(imgs["NE"], imgs["N"])


# ---------------------------------------------------------------- animate_asset

def test_animate_asset_returns_requested_frames(req):
    frames = service.animate_asset(req, frames=4)
    assert len(frames) == 4
    for f in frames:
        assert f.width == 32 and f.height == 32


def test_animate_asset_frames_share_identical_palette(req):
    frames = service.animate_asset(req, frames=4)
    palettes = {tuple(f.palette) for f in frames}
    assert len(palettes) == 1, "todos os frames devem ter exatamente a mesma subpaleta"
    palette = frames[0].palette
    for i, f in enumerate(frames):
        assert _colors(_decode(f.image_b64)) <= _hex_set(palette), f"frame {i}"


def test_animate_asset_frames_are_not_all_identical(req):
    frames = service.animate_asset(req, frames=4)
    imgs = [_decode(f.image_b64) for f in frames]
    assert any(not np.array_equal(imgs[0], other) for other in imgs[1:])


def test_animate_asset_clamps_frame_count(req):
    frames = service.animate_asset(req, frames=0)
    assert len(frames) == 1
