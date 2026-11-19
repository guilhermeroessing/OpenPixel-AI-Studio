"""Servicos puros: geracao de sprites, 8 direcoes, animacao, mapas e export de bundles."""
from __future__ import annotations

import base64
import io
import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import postprocess as pp  # importa o pacote -> sys.path de shared-contracts
from .providers import get_provider
from schemas import CompleteMapResponse, GenerateRequest, MapAsset, SpriteResponse  # noqa: E402

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


def _sprite(rgb: np.ndarray, palette: list[str], seed: int) -> SpriteResponse:
    return SpriteResponse(image_b64=_png_b64(rgb), width=rgb.shape[1], height=rgb.shape[0],
                          palette=palette, seed=seed, provider=_provider.name)


def _base(req: GenerateRequest):
    hr, seed = _provider.generate(req.prompt, _hr(req.size), req.seed)
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
