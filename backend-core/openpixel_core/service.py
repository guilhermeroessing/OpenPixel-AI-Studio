"""Servicos puros: geracao de sprites, 8 direcoes, animacao, mapas e export de bundles."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import animation
from . import postprocess as pp  # importa o pacote -> sys.path de shared-contracts
from .pixelate import pixelate
from .providers import get_provider
from schemas import (AnimateRequest, AnimationResponse, CompleteMapResponse, GenerateRequest, GenParams,  # noqa: E402
                     MapAsset, SpriteResponse)

_provider = get_provider()
DIRECTIONS = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
HR_SCALE = 8
MAP_W = MAP_H = 16
TILE = 32


def _hr(size: int) -> int:
    return min(1024, size * HR_SCALE)


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8), "RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _hex_to_rgb(palette: list[str]) -> np.ndarray:
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in palette], np.uint8)


def _png_b64_rgba(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, np.uint8), "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _sprite(rgb: np.ndarray, palette: list[str], seed: int) -> SpriteResponse:
    return SpriteResponse(image_b64=_png_b64(rgb), width=rgb.shape[1], height=rgb.shape[0],
                          palette=palette, seed=seed, provider=_provider.name)


# Fatores editaveis na interface: chave, rotulo, grupo, faixa, passo, ajuda. O padrao vem de default_params().
PARAM_SPECS = [
    {"key": "lora_pixel_art_xl", "label": "LoRA pixel-art-xl", "group": "Modelo", "min": 0, "max": 2, "step": 0.05,
     "help": "Força do estilo pixel art. Mais alto = pixels mais duros, proporções mais chibi."},
    {"key": "lora_redmond", "label": "LoRA PixelArtRedmond", "group": "Modelo", "min": 0, "max": 2, "step": 0.05,
     "help": "Força do estilo Redmond: proporções heroicas e mais detalhe. Muito alto fica ruidoso."},
    {"key": "cfg", "label": "CFG (obediência ao prompt)", "group": "Modelo", "min": 1, "max": 20, "step": 0.5,
     "help": "Mais alto segue o prompt mais à risca, mas satura as cores."},
    {"key": "steps", "label": "Passos de amostragem", "group": "Modelo", "min": 5, "max": 60, "step": 1,
     "help": "Mais passos = mais detalhe e mais tempo."},
    {"key": "pose_strength", "label": "Força da pose (ControlNet)", "group": "Animação", "min": 0, "max": 2,
     "step": 0.05, "help": "Quanto o personagem obedece ao esqueleto. Abaixo de ~0,8 os braços não seguem a pose."},
    {"key": "ref_weight", "label": "Referência do personagem (IP-Adapter)", "group": "Animação", "min": 0, "max": 2,
     "step": 0.05, "help": "Mantém o mesmo visual entre os frames. Alto demais deixa tudo chapado; 0 desliga."},
    {"key": "shoulder_spread", "label": "Separação dos ombros", "group": "Pose da corrida", "min": 0, "max": 0.1,
     "step": 0.005, "help": "Afasta o braço de trás do tronco (fração da altura)."},
    {"key": "flight_height", "label": "Altura do voo", "group": "Pose da corrida", "min": 0, "max": 0.1,
     "step": 0.005, "help": "Quanto o personagem sobe na fase de voo (fração da altura)."},
    {"key": "lean", "label": "Inclinação do tronco", "group": "Pose da corrida", "min": 0, "max": 0.7,
     "step": 0.025, "help": "Inclinação para a frente, em radianos. Maior = corrida mais agressiva."},
    {"key": "bg_tolerance", "label": "Tolerância do removedor de fundo", "group": "Pós-processo", "min": 10,
     "max": 150, "step": 5, "help": "Alto remove mais fundo (e sombra), mas pode comer partes do personagem."},
]


def default_params() -> dict[str, float]:
    d = {"lora_pixel_art_xl": 0.7, "lora_redmond": 0.7, "cfg": 6.5, "steps": 25, "pose_strength": 1.0,
         "ref_weight": 0.5, "shoulder_spread": 0.05, "flight_height": 0.025, "lean": 0.35, "bg_tolerance": 45}
    p = _provider
    if hasattr(p, "with_overrides"):  # ComfyProvider: o padrao real vem da instancia
        d.update(cfg=p.cfg, steps=p.steps, pose_strength=p.pose_strength, ref_weight=p.ref_weight)
        for name, strength in p.loras:
            low = name.lower()
            key = "lora_pixel_art_xl" if "pixel-art-xl" in low else "lora_redmond" if "redmond" in low else None
            if key:
                d[key] = strength
    return d


def _provider_for(params: GenParams | None):
    """Provider com as forcas da requisicao aplicadas (so o ComfyProvider suporta; o mock ignora)."""
    if params is None or not hasattr(_provider, "with_overrides"):
        return _provider
    lora = {k: v for k, v in (("pixel-art-xl", params.lora_pixel_art_xl), ("redmond", params.lora_redmond))
            if v is not None}
    return _provider.with_overrides(cfg=params.cfg, steps=params.steps, pose_strength=params.pose_strength,
                                    ref_weight=params.ref_weight, lora_strengths=lora or None)


def _base(req: GenerateRequest):
    hr, seed = _provider_for(req.params).generate(req.prompt, _hr(req.size), req.seed)
    # Padrao: media por bloco + paleta K-Means (openpixel_core.pixelate). PIXELIZER=pixeloe volta ao
    # pipeline antigo; seamless/symmetric/dither sempre usam o pipeline antigo (so ele os implementa).
    if os.environ.get("PIXELIZER", "box") == "box" and not (req.seamless or req.symmetric or req.dither):
        rgb = pixelate(hr, grid=req.size, colors=req.palette_size, out_size=req.size)
        pal = [pp._hex(c) for c in np.unique(rgb.reshape(-1, 3), axis=0)]
    else:
        rgb, pal = pp.pixelize(hr, req.size, req.palette_size, req.dither, req.seamless, req.symmetric)
    return rgb, pal, seed


def generate_pixel_art(req: GenerateRequest) -> SpriteResponse:
    rgb, pal, seed = _base(req)
    return _sprite(rgb, pal, seed)


def _lock(rgb: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    return pal_rgb[pp.quantize_to_palette(rgb, pal_rgb)].astype(np.uint8)


def _warp(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return cv2.warpAffine(np.ascontiguousarray(rgb), m, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)


def _dim(rgb: np.ndarray, f: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)


def generate_8_directions(req: GenerateRequest) -> dict[str, SpriteResponse]:
    """Uma base pixelizada; direcoes derivadas por transformacoes, travadas na mesma paleta."""
    base, pal, seed = _base(req)
    pal_rgb = _hex_to_rgb(pal)
    n = req.size
    flip = base[:, ::-1]
    front = base.copy()
    front[:, n - n // 2:] = front[:, :n // 2][:, ::-1]
    sq = 0.8
    m = np.array([[sq, 0, n * (1 - sq) / 2], [0, 1, 0]], np.float32)
    views = {
        "E": base, "W": flip, "S": front, "N": _dim(front, 0.8),
        "SE": _warp(base, m), "SW": _warp(flip, m),
        "NE": _warp(_dim(base, 0.85), m), "NW": _warp(_dim(flip, 0.85), m),
    }
    return {d: _sprite(_lock(views[d], pal_rgb), pal, seed) for d in DIRECTIONS}


def animate_asset(req: GenerateRequest, frames: int = 4) -> list[SpriteResponse]:
    """Ciclo de bob/sway; todos os frames usam exatamente a mesma subpaleta."""
    frames = max(1, frames)
    base, pal, seed = _base(req)
    pal_rgb = _hex_to_rgb(pal)
    amp = max(1.0, req.size / 16)
    out = []
    for i in range(frames):
        t = 2 * math.pi * i / frames
        dy, dx = round(math.sin(t) * amp), round(math.sin(2 * t) * amp / 2)
        sy = 1 + 0.05 * math.cos(t)
        m = np.array([[1, 0, dx], [0, sy, dy + req.size * (1 - sy)]], np.float32)
        out.append(_sprite(_lock(_warp(base, m), pal_rgb), pal, seed))
    return out


# ---------------------------------------------------------------- mapas

_TILE_LIB = [
    (("grass", "grama", "forest", "floresta", "meadow", "campo"), "grass", "seamless grass tile"),
    (("water", "agua", "lake", "lago", "river", "rio", "sea", "ocean"), "water", "seamless water tile"),
    (("sand", "areia", "desert", "deserto", "beach", "praia"), "sand", "seamless sand tile"),
    (("snow", "neve", "ice", "gelo"), "snow", "seamless snow tile"),
    (("stone", "pedra", "cave", "caverna", "dungeon", "masmorra"), "stone", "seamless stone floor tile"),
    (("lava", "volcan"), "lava", "seamless lava tile"),
]
_PROP_LIB = [
    (("tree", "arvore", "forest", "floresta"), "tree", "pixel art tree"),
    (("rock", "pedra", "stone", "cave"), "rock", "pixel art rock"),
    (("house", "casa", "village", "vila", "town"), "house", "pixel art small house"),
    (("chest", "bau", "treasure", "tesouro", "dungeon"), "chest", "pixel art treasure chest"),
    (("flower", "flor", "meadow"), "flower", "pixel art flower"),
    (("cactus", "desert", "deserto"), "cactus", "pixel art cactus"),
    (("torch", "tocha", "dungeon", "cave"), "torch", "pixel art torch"),
]


def _decompose(prompt: str) -> list[MapAsset]:
    low = prompt.lower()
    tiles = [MapAsset(name=n, kind="tile", prompt=f"{p}, {prompt}", size=TILE)
             for kw, n, p in _TILE_LIB if any(w in low for w in kw)]
    if not tiles:
        tiles = [MapAsset(name="grass", kind="tile", prompt=f"seamless grass tile, {prompt}", size=TILE)]
    if len(tiles) == 1:
        tiles.append(MapAsset(name="path", kind="tile", prompt="seamless dirt path tile", size=TILE))
    props = [MapAsset(name=n, kind="prop", prompt=f"{p}, {prompt}", size=TILE)
             for kw, n, p in _PROP_LIB if any(w in low for w in kw)]
    if not props:
        props = [MapAsset(name="rock", kind="prop", prompt=f"pixel art rock, {prompt}", size=TILE)]
    return tiles[:4] + props[:4]


def _tilemap(manifest: list[MapAsset], seed: int) -> dict[str, list[list[int]]]:
    rng = np.random.default_rng(seed)
    ids = {a.name: i + 1 for i, a in enumerate(manifest)}
    tiles = [a for a in manifest if a.kind == "tile"]
    props = [a for a in manifest if a.kind == "prop"]
    noise = cv2.resize(rng.random((4, 4)).astype(np.float32), (MAP_W, MAP_H), interpolation=cv2.INTER_CUBIC)
    sel = np.clip((noise * len(tiles)).astype(int), 0, len(tiles) - 1)
    ground = np.vectorize(lambda k: ids[tiles[k].name])(sel)
    obj = np.zeros((MAP_H, MAP_W), int)
    for _ in range(MAP_W * MAP_H // 10):
        x, y = int(rng.integers(0, MAP_W)), int(rng.integers(0, MAP_H))
        obj[y, x] = ids[props[int(rng.integers(0, len(props)))].name]
    return {"ground": ground.tolist(), "objects": obj.tolist(), "collision": (obj > 0).astype(int).tolist()}


def _render_composite(manifest, layers, imgs: dict[str, np.ndarray]) -> np.ndarray:
    by_id = {i + 1: a for i, a in enumerate(manifest)}
    canvas = np.zeros((MAP_H * TILE, MAP_W * TILE, 3), np.uint8)
    for layer in ("ground", "objects"):
        for y, row in enumerate(layers[layer]):
            for x, tid in enumerate(row):
                if not tid:
                    continue
                a = by_id[tid]
                spr = imgs[a.name]
                region = canvas[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE]
                if a.kind == "prop":
                    mask = (spr != spr[0, 0]).any(-1)
                    region[mask] = spr[mask]
                else:
                    region[:] = spr
    return canvas


def generate_complete_map(prompt: str, out_dir: str) -> CompleteMapResponse:
    out = Path(out_dir)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    manifest = _decompose(prompt)
    seed = int.from_bytes(prompt.encode()[:4].ljust(4, b"\0"), "little") % 100000
    raw = {}
    for a in manifest:
        hr, _ = _provider.generate(a.prompt, _hr(a.size), seed)
        raw[a.name] = pp.pixelize(hr, a.size, 8, False, seamless=(a.kind == "tile"), symmetric=False)
    all_rgb = np.concatenate([_hex_to_rgb(p) for _, p in raw.values()]).reshape(-1, 1, 3)
    master = pp.kmeans_palette(all_rgb, 24)
    master_hex = [pp._hex(c) for c in master]
    imgs, sprites = {}, {}
    for a in manifest:
        locked = _lock(raw[a.name][0], master)
        imgs[a.name] = locked
        used = sorted({pp._hex(c) for c in locked.reshape(-1, 3)})
        sprites[a.name] = _sprite(locked, used, seed)
        Image.fromarray(locked, "RGB").save(out / "assets" / f"{a.name}.png")
    layers = _tilemap(manifest, seed)
    comp_path = out / "composite_map.png"
    Image.fromarray(_render_composite(manifest, layers, imgs), "RGB").save(comp_path)
    manual = _manual(prompt, manifest, master_hex)
    manual_path = out / "MAP_ASSEMBLY_MANUAL.md"
    manual_path.write_text(manual, encoding="utf-8")
    (out / "map.json").write_text(json.dumps({
        "prompt": prompt, "tile_size": TILE, "width": MAP_W, "height": MAP_H,
        "master_palette": master_hex, "asset_manifest": [a.model_dump() for a in manifest],
        "tilemap_layers": layers}, indent=2), encoding="utf-8")
    return CompleteMapResponse(master_palette=master_hex, asset_manifest=manifest, tilemap_layers=layers,
                               composite_map_path=str(comp_path), manual_path=str(manual_path),
                               manual_markdown=manual, assets=sprites)


def _manual(prompt: str, manifest: list[MapAsset], palette: list[str]) -> str:
    rows = "\n".join(f"| {i + 1} | {a.name} | {a.kind} | assets/{a.name}.png |" for i, a in enumerate(manifest))
    return f"""# MAP_ASSEMBLY_MANUAL

Mapa gerado para o prompt: **{prompt}**

- Grade: {MAP_W}x{MAP_H} tiles de {TILE}x{TILE}px (composite: {MAP_W * TILE}x{MAP_H * TILE}px)
- Paleta mestre ({len(palette)} cores): {" ".join(palette)}
- Camadas: `ground` (tile base), `objects` (props, 0 = vazio), `collision` (1 = bloqueado)

## Assets (o ID e o valor usado nas camadas)

| ID | Nome | Tipo | Arquivo |
|----|------|------|---------|
{rows}

## Godot 4

1. Copie a pasta `assets/` para `res://`.
2. Crie um `TileSet` com tile size {TILE}x{TILE} e adicione cada PNG de tipo `tile` como atlas source.
3. Crie um `TileMapLayer` para `ground` e preencha com `godot_map.json` (`layers.ground`).
4. Props: instancie `Sprite2D` em `layers.objects` na posicao (x*{TILE}, y*{TILE}); use `layers.collision` para `StaticBody2D`.
5. Texturas: Filter = Nearest.

## Unity

1. Importe `assets/*.png` com Pixels Per Unit = {TILE}, Filter Mode = Point, Compression = None.
2. Crie `Tile` assets e uma `Tile Palette`.
3. Em Grid > Tilemap `Ground`, pinte conforme `unity_map.json` (`layers.ground`); props em Tilemap `Objects`.
4. Adicione `TilemapCollider2D` ao Tilemap `Objects`.

## Conferencia

Compare o resultado com `composite_map.png`.
"""


def export_engine_bundle(map_dir: str, engine: str) -> str:
    engine = engine.lower()
    if engine not in ("godot", "unity"):
        raise ValueError("engine deve ser 'godot' ou 'unity'")
    d = Path(map_dir)
    data = json.loads((d / "map.json").read_text(encoding="utf-8"))
    sources = [{"id": i, "name": a["name"], "kind": a["kind"], "texture": f"assets/{a['name']}.png"}
               for i, a in enumerate(data["asset_manifest"], 1)]
    if engine == "godot":
        doc = {"format": "openpixel-godot-4", "tile_size": [TILE, TILE], "map_size": [data["width"], data["height"]],
               "tileset_sources": sources, "layers": data["tilemap_layers"], "empty_cell": 0}
        name = "godot_map.json"
    else:
        doc = {"format": "openpixel-unity", "cellSize": [1, 1, 0], "pixelsPerUnit": TILE,
               "width": data["width"], "height": data["height"], "tiles": sources, "layers": data["tilemap_layers"]}
        name = "unity_map.json"
    zpath = d / f"{d.name}_{engine}_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, json.dumps(doc, indent=2))
        for f in sorted((d / "assets").glob("*.png")):
            z.write(f, f"assets/{f.name}")
        for extra in ("composite_map.png", "MAP_ASSEMBLY_MANUAL.md"):
            if (d / extra).exists():
                z.write(d / extra, extra)
    return str(zpath)


# ---------------------------------------------------------------- animacao de corrida

def animate_run(req: AnimateRequest, progress=None) -> AnimationResponse:
    """Animacao por movimento (corrida, idle...): ComfyUI (pose + referencia) ou, sem ele, um balanco simples."""
    p = req.params or GenParams()
    motion = animation.MOTIONS[req.motion]
    n_frames = req.frames or motion["frames"]
    seed = req.seed if req.seed is not None else animation.new_seed()
    prov = _provider_for(req.params)
    if not hasattr(prov, "with_overrides"):
        frames = animate_asset(req.model_copy(update={"seed": seed}), n_frames)
        if progress:
            progress(len(frames), len(frames))
        return AnimationResponse(frames=frames, seed=seed, fps=motion["fps"], provider=_provider.name)
    defaults = default_params()
    if not prov.avoid:
        prov = prov.with_overrides(avoid=animation.avoid_for(motion))
    shape = {k: (getattr(p, k) if getattr(p, k) is not None else defaults[k])
             for k in ("shoulder_spread", "lean", "flight_height")}
    raws = animation.generate_motion_raws(
        prov, req.prompt, prov.native_size, seed, motion["poses"](n_frames, shape), motion["suffix"],
        prov.ref_weight > 0, progress, motion.get("prompt_weight", 1.0))
    rgba = animation.process_frames(raws, req.size, req.palette_size,
                                    p.bg_tolerance if p.bg_tolerance is not None else defaults["bg_tolerance"],
                                    anchor=motion["anchor"])
    out = []
    for a in rgba:
        used = sorted({pp._hex(c) for c in a[a[..., 3] > 0][:, :3]})
        out.append(SpriteResponse(image_b64=_png_b64_rgba(a), width=req.size, height=req.size, palette=used,
                                  seed=seed, provider=_provider.name))
    return AnimationResponse(frames=out, seed=seed, fps=motion["fps"], provider=_provider.name)
