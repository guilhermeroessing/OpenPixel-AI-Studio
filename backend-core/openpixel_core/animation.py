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
from .poses import idle_cycle, punch_cycle, render_pose, run_cycle, skeleton, sword_cycle

RUN_SUFFIX = ("running, three-quarter side view facing right, arms swinging, both arms visible, "
              "full body, simple flat background")
PUNCH_SUFFIX = ("throwing a punch, fist extended forward, fighting stance, bare hands, three-quarter side view "
               "facing right, both arms visible, full body, simple flat background")
SWORD_SUFFIX = ("two-handed overhead downward chop, both bare hands clasped together raised then swung down in front, empty hands, "
                "three-quarter side view facing right, both arms visible, full body, simple flat background")
REF_SUFFIX = ("standing relaxed, three-quarter side view facing right, arms hanging at the sides, both arms visible, "
              "full body, simple flat background")
IDLE_SUFFIX = ("standing idle, relaxed, breathing, three-quarter side view facing right, arms hanging at the sides, "
               "both arms visible, full body, simple flat background")

# Movimentos disponiveis: texto acrescentado ao prompt, fps do loop e gerador das poses (quadros x parametros).
MOTIONS = {
    "run": {"label": "Corrida", "fps": 10, "suffix": RUN_SUFFIX, "frames": 8, "anchor": "centroid",
            "poses": lambda frames, p: run_cycle(frames, half=True, shoulder_spread=p.get("shoulder_spread", 0.05),
                                                 lean=p.get("lean", 0.35), flight=p.get("flight_height", 0.025))},
    "punch": {"label": "Soco", "fps": 10, "suffix": PUNCH_SUFFIX, "frames": 6, "anchor": "fixed", "prompt_weight": 1.3,
              "avoid": "(boxing gloves:1.5), (gloves:1.4), smoke, dust, aura, energy effect, particles, motion blur, "
                       "speed lines, glow, (ground shadow:1.5), (floor:1.3)",
              "poses": lambda frames, p: punch_cycle(frames)},
    "sword": {"label": "Golpe de espada", "fps": 10, "suffix": SWORD_SUFFIX, "frames": 6, "anchor": "fixed",
              "avoid": "(sword:1.6), (katana:1.5), (blade:1.5), (weapon:1.5), (knife:1.4), (stick:1.3), (staff:1.3), (boxing gloves:1.5), (red gloves:1.5), "
                       "smoke, dust, aura, energy effect, slash effect, motion blur, speed lines, glow, "
                       "(ground shadow:1.5), (floor:1.3)",
              "poses": lambda frames, p: sword_cycle(frames, reach=1.5)},
    "idle": {"label": "Parado (idle)", "fps": 4, "suffix": IDLE_SUFFIX, "frames": 4, "anchor": "fixed",
             "poses": lambda frames, p: idle_cycle(frames)},
}
SCENE_AVOID = "close-up, bust, cropped, multiple views"
RUN_AVOID = "(weapon:1.6), (shield:1.8), (sword:1.5), (spear:1.5), (gun:1.5), " + SCENE_AVOID


def _components(mask: np.ndarray, connectivity: int = 8):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    return n, lab, stats


def _torso_protect(pose, g: int, radius: float = 0.045) -> np.ndarray:
    """Faixa em volta do tronco da pose (pescoco -> meio do quadril): sempre e personagem, mesmo que a roupa tenha
    a cor do fundo (regata branca em fundo claro)."""
    m = np.zeros((g, g), np.uint8)
    neck = pose[1]
    hip = ((pose[8][0] + pose[11][0]) / 2, (pose[8][1] + pose[11][1]) / 2)
    cv2.line(m, (round(neck[0] * (g - 1)), round(neck[1] * (g - 1))), (round(hip[0] * (g - 1)), round(hip[1] * (g - 1))),
             1, max(1, round(2 * radius * g)))
    return m.astype(bool)


def _remove_bg(a: np.ndarray, tol: int, clean_shadow: bool = True, protect: np.ndarray | None = None,
               leak_guard: bool = False) -> np.ndarray:
    """Mascara do personagem (True) em um frame pixelizado: tira o fundo ligado as bordas e a sombra do chao.

    ``protect``: mascara de pixels que nunca sao fundo (o tronco, vindo da pose)."""
    g = a.shape[0]
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    vals, counts = np.unique(edge.reshape(-1, 3), axis=0, return_counts=True)
    bg = vals[counts.argmax()].astype(int)
    near = np.abs(a.astype(int) - bg).sum(axis=2) < tol
    n, lab, _ = _components(near, 4)
    border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    bgmask = np.isin(lab, list(border))
    if leak_guard:
        # abertura: fresta de 1-2 px entre o corpo e a roupa branca nao deixa o fundo "entrar" e apagar a roupa
        core = cv2.erode(near.astype(np.uint8), np.ones((3, 3), np.uint8))
        _, lab2, _ = _components(core, 4)
        edge2 = set(lab2[0]) | set(lab2[-1]) | set(lab2[:, 0]) | set(lab2[:, -1])
        edge2.discard(0)
        grown = cv2.dilate(np.isin(lab2, list(edge2)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        bgmask = bgmask & grown
    fg = ~bgmask
    if protect is not None:
        fg |= protect
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


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def palette_is_tonal(palette: list[tuple[int, int, int]], max_chroma: float = 35.0) -> bool:
    """Paleta de tons (quase sem cor, ex.: azul-marinho + creme): so a luminosidade distingue as cores."""
    lab = _lab(np.asarray(palette, np.uint8).reshape(1, -1, 3)).reshape(-1, 3)
    return bool(np.hypot(lab[:, 1] - 128, lab[:, 2] - 128).max() < max_chroma)


def snap_to_palette(arr: np.ndarray, palette: list[tuple[int, int, int]],
                    src_range: tuple[float, float] | None = None) -> np.ndarray:
    """Cada pixel vira a cor da paleta mais proxima em Lab (distancia perceptual).

    Paletas de tons (``palette_is_tonal``) usam so a luminosidade, esticada de ``src_range`` (faixa de L do
    personagem) para a faixa da paleta: sem isso, uma paleta de 2 cores joga quase tudo na mais clara.
    """
    pal = np.asarray(palette, np.uint8).reshape(1, -1, 3)
    lab_pal = _lab(pal).reshape(-1, 3)
    lab = _lab(arr).reshape(-1, 1, 3)
    if src_range is not None and palette_is_tonal(palette):
        lo, hi = src_range
        lmin, lmax = float(lab_pal[:, 0].min()), float(lab_pal[:, 0].max())
        stretched = np.clip((lab[:, 0, 0] - lo) / max(hi - lo, 1e-3), 0, 1) * (lmax - lmin) + lmin
        idx = np.abs(stretched[:, None] - lab_pal[None, :, 0]).argmin(axis=1)
    else:
        idx = ((lab - lab_pal[None]) ** 2).sum(axis=2).argmin(axis=1)
    return pal.reshape(-1, 3)[idx].reshape(arr.shape).astype(np.uint8)


def process_frames(raws: list[np.ndarray], grid: int = 128, colors: int = 24, bg_tol: int = 45,
                   anchor: str = "centroid", poses: list | None = None, leak_guard: bool = True,
                   custom_palette: list[tuple[int, int, int]] | None = None) -> list[np.ndarray]:
    """Frames brutos (RGB alta resolucao) -> RGBA grid x grid: paleta unica, sem fundo, alinhados.

    anchor: "centroid" recentraliza cada frame na horizontal (bom para corrida, em que o corpo avanca e recua);
    "fixed" aplica o MESMO deslocamento a todos os frames (local fixo: o que se move e so o movimento da pose,
    ideal para idle e ataques parados).

    custom_palette: cores escolhidas pelo usuario (2 a 8). O fundo e achado com a paleta automatica (para nao
    se misturar com a roupa) e so depois o personagem e repintado, pixel a pixel, com as cores escolhidas.
    """
    if anchor not in ("centroid", "fixed"):
        raise ValueError("anchor deve ser 'centroid' ou 'fixed'")
    small = [np.asarray(Image.fromarray(r).convert("RGB").resize((grid, grid), Image.BOX)) for r in raws]
    q = Image.fromarray(np.concatenate(small, axis=1)).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, kmeans=4, dither=Image.Dither.NONE)
    pal = [tuple(int(v) for v in c) for c in np.array(q.getpalette()[:colors * 3], np.uint8).reshape(-1, 3)]
    items = []
    for i, r in enumerate(raws):
        a = pixelate(r, grid, palette=pal, out_size=None)
        mask = _remove_bg(a, bg_tol, protect=_torso_protect(poses[i], grid) if poses else None, leak_guard=leak_guard)
        ys, xs = np.nonzero(mask)
        items.append((a, mask, ys, xs))
    if custom_palette:
        # luminosidade do personagem em TODOS os frames (mesma faixa para todos: sem piscar de um frame para outro)
        lum = np.concatenate([_lab(small[i])[it[1], 0] for i, it in enumerate(items)])
        src = (float(np.percentile(lum, 5)), float(np.percentile(lum, 95)))
        items = [(snap_to_palette(small[i], custom_palette, src), m, ys, xs) for i, (_, m, ys, xs) in enumerate(items)]
    margin = max(1, round(grid * 6 / 128))
    dy = (grid - margin) - max(it[2].max() for it in items)  # um unico dy: preserva a subida do voo
    fixed_dx = grid // 2 - int(round(float(np.mean([it[3].mean() for it in items]))))
    out = []
    for a, mask, ys, xs in items:
        dx = fixed_dx if anchor == "fixed" else grid // 2 - int(round(xs.mean()))
        canvas = np.zeros((grid, grid, 4), np.uint8)
        sy, sx = ys + dy, xs + dx
        ok = (sy >= 0) & (sy < grid) & (sx >= 0) & (sx < grid)
        canvas[sy[ok], sx[ok], :3] = a[ys[ok], xs[ok]]
        canvas[sy[ok], sx[ok], 3] = 255
        out.append(canvas)
    return out


def generate_motion_raws(provider, prompt: str, native: int, seed: int, poses: list, suffix: str,
                         use_reference: bool, progress: Callable[[int, int], None] | None = None,
                         prompt_weight: float = 1.0) -> list[np.ndarray]:
    """Gera uma imagem bruta por pose (mesma seed; a primeira pose vira a referencia do personagem).

    prompt_weight > 1 da mais peso ao texto do personagem do que ao do movimento (evita que "soco" pinte tudo de vermelho).
    """
    who = prompt.strip() if prompt_weight == 1.0 else f"({prompt.strip()}:{prompt_weight})"
    text = f"{who}, {suffix}"
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
            # a referencia do personagem e neutra (em pe, sem o texto do movimento): "soco" nao pinta a roupa de vermelho
            # e as cores certas passam, via IP-Adapter, para todos os frames
            neutral = Path(tmp) / "pose_ref.png"
            Image.fromarray(render_pose(skeleton(), native, 1.5)).save(neutral)
            ref0, _ = provider.generate(f"{who}, {REF_SUFFIX}", native, seed, pose=str(neutral))
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


def avoid_for(motion: dict) -> str:
    """Prompt negativo do movimento: o geral (armas, vistas multiplas) mais os termos proprios dele."""
    base = motion.get("base_avoid", RUN_AVOID)
    extra = motion.get("avoid")
    return f"{base}, {extra}" if extra else base


def custom_motion(poses: list, text: str | None = None, fps: int | None = None, anchor: str | None = None,
                  allow_objects: bool = False, extra_avoid: str | None = None) -> dict:
    """Movimento montado pelo usuario (editor de poses): as poses vem prontas e o texto descreve o gesto.

    Por padrao proibe armas (o gesto e do corpo); ``allow_objects`` tira essa proibicao, por conta de quem usa.
    """
    text = (text or "").strip().rstrip(",")
    suffix = f"{text}, full body, simple flat background" if text else "full body, simple flat background"
    motion = {"label": "Personalizado", "fps": fps or 8, "suffix": suffix, "frames": len(poses),
              "anchor": anchor or "fixed", "poses": lambda frames, p: poses}
    if allow_objects:
        motion["base_avoid"] = SCENE_AVOID
    if extra_avoid and extra_avoid.strip():
        motion["avoid"] = extra_avoid.strip()
    return motion


def new_seed() -> int:
    return random.randint(1, 10**6)
