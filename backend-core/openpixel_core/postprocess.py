"""Pos-processamento pixel art: downscale, K-Means CIELAB, ilhas, seamless, simetria, dither."""
from __future__ import annotations

import cv2
import numpy as np

_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16 - 0.5


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def _hex(c) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in c])


def make_seamless(rgb: np.ndarray) -> np.ndarray:
    """Cross-fade com versao rolada em meia largura: bordas opostas passam a casar."""
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float32)
    rolled = np.roll(f, (h // 2, w // 2), axis=(0, 1))
    wy = np.sin(np.pi * (np.arange(h) + 0.5) / h) ** 2
    wx = np.sin(np.pi * (np.arange(w) + 0.5) / w) ** 2
    m = (wy[:, None] * wx[None, :])[..., None]
    return np.clip(f * m + rolled * (1 - m), 0, 255).astype(np.uint8)


def downscale(rgb: np.ndarray, size: int) -> np.ndarray:
    """PixelOE se disponivel; senao fallback box (INTER_AREA) -> NEAREST."""
    try:
        from pixeloe.pixelize import pixelize as _pe  # type: ignore

        out = _pe(rgb, target_size=size, patch_size=max(2, rgb.shape[0] // size))
        out = np.asarray(out)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if out.shape[:2] != (size, size):
            out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
        return out
    except Exception:
        pass
    h, w = rgb.shape[:2]
    if h < size or w < size:
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def kmeans_palette(rgb: np.ndarray, k: int) -> np.ndarray:
    """Paleta (k,3 uint8 RGB) via K-Means em CIELAB."""
    lab = _to_lab(rgb).reshape(-1, 3)
    uniq = np.unique(lab, axis=0)
    k = max(1, min(k, len(uniq)))
    if k == len(uniq):
        centers = uniq
    else:
        cv2.setRNGSeed(0)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    c8 = np.clip(centers, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(c8, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def quantize_to_palette(rgb: np.ndarray, palette: np.ndarray, dither: bool = False) -> np.ndarray:
    """Mapeia para a paleta fixa (CIELAB). Retorna mapa de indices (H,W)."""
    pal = np.asarray(palette, np.uint8).reshape(-1, 3)
    lab = _to_lab(rgb)
    plab = _to_lab(pal.reshape(-1, 1, 3)).reshape(-1, 3)
    if dither and len(pal) > 1:
        d = np.sqrt(((plab[:, None] - plab[None]) ** 2).sum(-1))
        d[d == 0] = np.inf
        step = float(np.median(d.min(1)))
        h, w = lab.shape[:2]
        t = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        lab = lab + (t * step * 0.8)[..., None]
    dist = ((lab[:, :, None, :] - plab[None, None]) ** 2).sum(-1)
    return dist.argmin(-1).astype(np.int32)


def remove_islands(idx: np.ndarray, palette: np.ndarray, max_area: int = 2, wrap: bool = False) -> np.ndarray:
    """Reatribui componentes conexos (4-viz.) de <= max_area px a cor vizinha dominante."""
    idx = idx.copy()
    h, w = idx.shape
    pad = 2 if wrap else 0
    if wrap:
        idx = np.pad(idx, pad, mode="wrap")
    for _ in range(2):
        changed = False
        for lab_id in np.unique(idx):
            mask = (idx == lab_id).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] > max_area:
                    continue
                comp = (cc == i).astype(np.uint8)
                ring = cv2.dilate(comp, np.ones((3, 3), np.uint8)) - comp
                neigh = idx[ring.astype(bool)]
                neigh = neigh[neigh != lab_id]
                if neigh.size:
                    idx[comp.astype(bool)] = np.bincount(neigh).argmax()
                    changed = True
        if not changed:
            break
    if wrap:
        idx = idx[pad:pad + h, pad:pad + w]
    return idx


def pixelize(rgb: np.ndarray, size: int, palette_size: int = 16, dither: bool = False,
             seamless: bool = False, symmetric: bool = False, palette: np.ndarray | None = None):
    """rgb alta resolucao -> (RGB uint8 size x size, paleta hex). `palette` opcional trava a paleta."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    if seamless:
        rgb = make_seamless(rgb)
    small = downscale(rgb, size)
    if symmetric:
        half = size // 2
        small = small.copy()
        small[:, size - half:] = small[:, :half][:, ::-1]
    pal = kmeans_palette(small, palette_size) if palette is None else np.asarray(palette, np.uint8).reshape(-1, 3)
    idx = quantize_to_palette(small, pal, dither)
    if not dither:
        idx = remove_islands(idx, pal, 2, wrap=seamless)
    out = pal[idx]
    used = sorted(set(idx.ravel().tolist()))
    return out.astype(np.uint8), [_hex(pal[i]) for i in used]
