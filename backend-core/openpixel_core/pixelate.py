"""Pixelizacao real: reduz para uma grade pequena, quantiza a paleta e amplia sem suavizar.

O SDXL gera "estilo pixel art" em 1024 px, mas os pixels nao sao alinhados a uma grade.
Este modulo converte o resultado em pixel art de verdade (grade fixa + paleta limitada).

Paleta:
  - ``palette=None``  -> paleta adaptativa de ``colors`` cores (median-cut refinado por k-means);
  - ``palette=[(r,g,b), ...]`` -> paleta fixa (ex.: Endesga 32, DawnBringer 32): cada pixel vira a cor mais proxima.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _downscale(im: Image.Image, grid: int, method: str) -> Image.Image:
    if method == "box":  # media de cada bloco: suave, mantem a cor dominante
        return im.resize((grid, grid), Image.BOX)
    if method == "mode":  # cor mais frequente (apos quantizar) de cada bloco: bordas mais nitidas
        q = im.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        idx = np.asarray(q)
        h, w = idx.shape
        ys = np.linspace(0, h, grid + 1).astype(int)
        xs = np.linspace(0, w, grid + 1).astype(int)
        pal = np.asarray(q.getpalette(), dtype=np.uint8).reshape(-1, 3)
        out = np.zeros((grid, grid, 3), dtype=np.uint8)
        for y in range(grid):
            for x in range(grid):
                tile = idx[ys[y]:ys[y + 1], xs[x]:xs[x + 1]].ravel()
                out[y, x] = pal[np.bincount(tile).argmax()]
        return Image.fromarray(out)
    raise ValueError(f"method desconhecido: {method!r} (use 'box' ou 'mode')")


def _snap_to_palette(arr: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    pal = np.asarray(palette, dtype=np.int32)
    flat = arr.reshape(-1, 1, 3).astype(np.int32)
    idx = ((flat - pal[None]) ** 2).sum(axis=2).argmin(axis=1)
    return pal[idx].reshape(arr.shape).astype(np.uint8)


def pixelate(img: np.ndarray | Image.Image, grid: int = 128, colors: int = 24,
             out_size: int | None = 1024, palette: list[tuple[int, int, int]] | None = None,
             method: str = "box", flat_bg: bool = True) -> np.ndarray:
    """Retorna RGB uint8 (out_size x out_size) com ``grid`` x ``grid`` pixels logicos.

    flat_bg: se True, o fundo (cor mais comum nas bordas) vira uma cor unica, sem ruido.
    """
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    small = _downscale(im.convert("RGB"), grid, method)
    if palette:
        arr = _snap_to_palette(np.asarray(small, dtype=np.uint8), palette)
    else:
        # kmeans=4 refina a paleta do median-cut: preserva melhor cores de destaque (ex.: laranja, ciano)
        q = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, kmeans=4, dither=Image.Dither.NONE)
        arr = np.asarray(q.convert("RGB"), dtype=np.uint8).copy()
    if flat_bg:
        edge = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        vals, counts = np.unique(edge.reshape(-1, 3), axis=0, return_counts=True)
        bg = vals[counts.argmax()]
        arr[(np.abs(arr.astype(int) - bg.astype(int)).sum(axis=2) < 24)] = bg
    out = Image.fromarray(arr)
    if out_size:
        out = out.resize((out_size, out_size), Image.NEAREST)
    return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))
