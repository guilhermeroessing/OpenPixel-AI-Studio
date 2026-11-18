"""OpenPixel AI Gateway: servidor MCP (FastMCP) que expõe o backend-core como tools."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "shared-contracts", _ROOT / "backend-core", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("OpenPixel")


def _service():
    from openpixel_core import service

    return service


def _request(prompt: str, **fields):
    from schemas import GenerateRequest

    return GenerateRequest(prompt=prompt, **{k: v for k, v in fields.items() if v is not None})


@mcp.tool
def generate_pixel_art(
    prompt: str,
    size: int = 32,
    kind: str = "asset",
    palette_size: int = 16,
    seamless: bool = False,
    symmetric: bool = False,
    dither: bool = False,
    seed: int | None = None,
) -> dict:
    """Gera um sprite pixel art a partir de um prompt."""
    req = _request(prompt, size=size, kind=kind, palette_size=palette_size,
                   seamless=seamless, symmetric=symmetric, dither=dither, seed=seed)
    return _service().generate_pixel_art(req).model_dump()


@mcp.tool
def generate_8_directions(
    prompt: str,
    size: int = 32,
    palette_size: int = 16,
    seed: int | None = None,
) -> dict:
    """Gera o personagem nas 8 direções; retorna {direção: sprite}."""
    req = _request(prompt, size=size, kind="character", palette_size=palette_size, seed=seed)
    result = _service().generate_8_directions(req)
    return {name: sprite.model_dump() for name, sprite in result.items()}


@mcp.tool
def animate_asset(
    prompt: str,
    frames: int = 4,
    size: int = 32,
    kind: str = "character",
    palette_size: int = 16,
    seed: int | None = None,
) -> list[dict]:
    """Anima um asset e retorna a lista de frames."""
    req = _request(prompt, size=size, kind=kind, palette_size=palette_size, seed=seed)
    return [f.model_dump() for f in _service().animate_asset(req, frames=frames)]


@mcp.tool
def generate_complete_map(prompt: str, out_dir: str = "output/map") -> dict:
    """Gera um mapa completo: pacote de assets, tilemap e manual MAP_ASSEMBLY_MANUAL.md.

    O texto do manual vem em `manual_markdown`.
    """
    result = _service().generate_complete_map(prompt, out_dir)
    data = result.model_dump()
    if not data.get("manual_markdown"):
        try:
            data["manual_markdown"] = Path(data["manual_path"]).read_text(encoding="utf-8")
        except (OSError, KeyError):
            data["manual_markdown"] = ""
    return data


@mcp.tool
def export_engine_bundle(map_dir: str, engine: str = "godot") -> dict:
    """Exporta o mapa gerado para um bundle de engine (ex.: godot, unity, tiled)."""
    path = _service().export_engine_bundle(map_dir, engine)
    return {"engine": engine, "bundle_path": str(path)}


if __name__ == "__main__":
    mcp.run()
