"""Animacao de corrida: poses (ControlNet) + referencia (IP-Adapter) + pos-processamento dos frames.

Pipeline (ComfyUI): gera uma imagem de referencia na pose 0, depois cada frame com a mesma seed, a pose do
ciclo e o IP-Adapter apontando para a referencia. O pos-processamento (process_frames) usa uma paleta unica
para todos os frames, tira o fundo, apaga a sombra fina do chao e alinha: um deslocamento vertical unico
(o personagem sobe sozinho no voo) e um horizontal por frame (centroide). Saida: RGBA com fundo transparente.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .pixelate import pixelate
from .poses import idle_cycle, render_pose, run_cycle

RUN_SUFFIX = ("running, three-quarter side view facing right, arms swinging, both arms visible, "
              "full body, simple flat background")
IDLE_SUFFIX = ("standing idle, relaxed, breathing, three-quarter side view facing right, arms hanging at the sides, "
               "both arms visible, full body, simple flat background")

# Movimentos disponiveis: texto acrescentado ao prompt, fps do loop e gerador das poses (quadros x parametros).
MOTIONS = {
    "run": {"label": "Corrida", "fps": 10, "suffix": RUN_SUFFIX,
            "poses": lambda frames, p: run_cycle(frames, half=True, shoulder_spread=p.get("shoulder_spread", 0.05),
                                                 lean=p.get("lean", 0.35), flight=p.get("flight_height", 0.025))},
    "idle": {"label": "Parado (idle)", "fps": 6, "suffix": IDLE_SUFFIX,
             "poses": lambda frames, p: idle_cycle(frames)},
}
RUN_AVOID = "(weapon:1.6), (shield:1.8), (sword:1.5), (spear:1.5), (gun:1.5), close-up, bust, cropped, multiple views"


def _components(mask: np.ndarray, connectivity: int = 8):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    return n, lab, stats


def _remove_bg(a: np.ndarray, tol: int, clean_shadow: bool = True) -> np.ndarray:
    """Mascara do personagem (True) em um frame pixelizado: tira o fundo ligado as bordas e a sombra do chao."""
    g = a.shape[0]
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    vals, counts = np.unique(edge.reshape(-1, 3), axis=0, return_counts=True)
    bg = vals[counts.argmax()].astype(int)
    near = np.abs(a.astype(int) - bg).sum(axis=2) < tol
    n, lab, _ = _components(near, 4)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    fg = ~np.isin(lab, list(border))
    if not fg.any():
        return np.ones((g, g), bool)
    # sombra fina no chao: faixas horizontais largas nas ultimas linhas, sem corpo logo acima
    ys = np.nonzero(fg.any(axis=1))[0]
    min_run = max(4, round(g * 0.2)) if clean_shadow else g + 1  # sombra: >= 20% da largura (tenis: ~11%)
    # de cima para baixo: ao apagar a linha de cima da faixa, a de baixo passa a ficar sem corpo acima (cascata)
    for y in range(max(ys.max() - max(3, round(g * 8 / 128)), 1), ys.max() + 1):
        x = 0
        while x < g:
            if fg[y, x]:
                x2 = x
                while x2 < g and fg[y, x2]:
                    x2 += 1
                if x2 - x >= min_run and fg[y - 1, x:x2].mean() < 0.6:
                    fg[y, x:x2] = False
                x = x2
            else:
                x += 1
    n, lab, stats = _components(fg, 8)
    if n <= 1:
        return np.ones((g, g), bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, ar in enumerate(areas) if ar >= max(4, 0.04 * areas.max())]  # braco solto nao some
    return np.isin(lab, keep)


def process_frames(raws: list[np.ndarray], grid: int = 128, colors: int = 24, bg_tol: int = 45) -> list[np.ndarray]:
    """Frames brutos (RGB alta resolucao) -> RGBA grid x grid: paleta unica, sem fundo, alinhados."""
    small = [np.asarray(Image.fromarray(r).convert("RGB").resize((grid, grid), Image.BOX)) for r in raws]
    q = Image.fromarray(np.concatenate(small, axis=1)).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, kmeans=4, dither=Image.Dither.NONE)
    pal = [tuple(int(v) for v in c) for c in np.array(q.getpalette()[:colors * 3], np.uint8).reshape(-1, 3)]
    items = []
    for r in raws:
        a = pixelate(r, grid, palette=pal, out_size=None)
        mask = _remove_bg(a, bg_tol)
        ys, xs = np.nonzero(mask)
        items.append((a, mask, ys, xs))
    margin = max(1, round(grid * 6 / 128))
    dy = (grid - margin) - max(it[2].max() for it in items)  # um unico dy: preserva a subida do voo
    out = []
    for a, mask, ys, xs in items:
        dx = grid // 2 - int(round(xs.mean()))
        canvas = np.zeros((grid, grid, 4), np.uint8)
        sy, sx = ys + dy, xs + dx
        ok = (sy >= 0) & (sy < grid) & (sx >= 0) & (sx < grid)
        canvas[sy[ok], sx[ok], :3] = a[ys[ok], xs[ok]]
        canvas[sy[ok], sx[ok], 3] = 255
        out.append(canvas)
    return out


def generate_motion_raws(provider, prompt: str, native: int, seed: int, poses: list, suffix: str,
                         use_reference: bool, progress: Callable[[int, int], None] | None = None) -> list[np.ndarray]:
    """Gera uma imagem bruta por pose (mesma seed; a primeira pose vira a referencia do personagem)."""
    text = f"{prompt.strip()}, {suffix}"
    total = len(poses) + (1 if use_reference else 0)
    raws: list[np.ndarray] = []
    done = 0
    with tempfile.TemporaryDirectory(prefix="openpixel_run_") as tmp:
        paths = []
        for i, k in enumerate(poses):
            pth = Path(tmp) / f"pose_{i}.png"
            Image.fromarray(render_pose(k, native, 1.5)).save(pth)
            paths.append(str(pth))
        ref = None
        if use_reference:
            ref0, _ = provider.generate(text, native, seed, pose=paths[0])
            ref = str(Path(tmp) / "ref.png")
            Image.fromarray(ref0).save(ref)
            done += 1
            if progress:
                progress(done, total)
        for pth in paths:
            img, _ = provider.generate(text, native, seed, pose=pth, reference=ref)
            raws.append(img)
            done += 1
            if progress:
                progress(done, total)
    return raws


def new_seed() -> int:
    return random.randint(1, 10**6)
