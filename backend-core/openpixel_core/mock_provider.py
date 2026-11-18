"""SmartMockProvider: simula a saida de uma IA com formas, ruido e degrades (CPU)."""
from __future__ import annotations

import hashlib
import random

import cv2
import numpy as np

_HUES = {
    "grass": 60, "grama": 60, "forest": 65, "floresta": 65, "tree": 62, "arvore": 62,
    "water": 105, "agua": 105, "lake": 105, "ocean": 108, "sea": 108,
    "sand": 20, "areia": 20, "desert": 18, "deserto": 18,
    "fire": 5, "lava": 3, "stone": 0, "pedra": 0, "snow": 100, "neve": 100,
}


def _hue_for(prompt: str, rng: np.random.Generator) -> int:
    low = prompt.lower()
    for k, h in _HUES.items():
        if k in low:
            return h
    return int(rng.integers(0, 180))


def _value_noise(rng: np.random.Generator, n: int, octaves: int = 4) -> np.ndarray:
    out = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        g = 4 * 2**o
        small = rng.random((g, g)).astype(np.float32)
        out += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    return out / tot


class SmartMockProvider:
    name = "smart_mock"

    def generate(self, prompt: str, size: int = 512, seed: int | None = None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng((seed * 1_000_003 + h) % 2**32)
        hue = _hue_for(prompt, rng)
        noise = _value_noise(rng, size)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        grad = 0.6 + 0.4 * yy
        hsv = np.zeros((size, size, 3), np.float32)
        hsv[..., 0] = (hue + 20 * (noise - 0.5)) % 180
        hsv[..., 1] = np.clip(0.45 + 0.4 * noise, 0, 1) * 255
        hsv[..., 2] = np.clip(grad * (0.55 + 0.45 * noise), 0, 1) * 255
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        # formas geometricas centrais
        for _ in range(int(rng.integers(2, 5))):
            c = (int(rng.integers(size // 4, 3 * size // 4)), int(rng.integers(size // 4, 3 * size // 4)))
            col = tuple(int(v) for v in rng.integers(30, 255, 3))
            r = int(rng.integers(size // 10, size // 4))
            if rng.random() < 0.5:
                cv2.circle(img, c, r, col, -1)
            else:
                cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col, -1)
        return np.ascontiguousarray(img, dtype=np.uint8), seed
