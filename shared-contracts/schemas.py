"""Contratos compartilhados (Pydantic) entre backend, gateway MCP e frontend."""
from typing import Literal
from pydantic import BaseModel, Field

AssetKind = Literal["character", "asset", "tileset", "map"]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    size: int = Field(default=32, ge=16, le=128)
    kind: AssetKind = "asset"
    palette_size: int = Field(default=16, ge=2, le=64)
    seamless: bool = False
    symmetric: bool = False
    dither: bool = False
    seed: int | None = None


class SpriteResponse(BaseModel):
    image_b64: str
    width: int
    height: int
    palette: list[str]
    seed: int
    provider: str = "smart_mock"


class MapAsset(BaseModel):
    name: str
    kind: Literal["tile", "prop"]
    prompt: str
    size: int = 32


class CompleteMapResponse(BaseModel):
    master_palette: list[str]
    asset_manifest: list[MapAsset]
    tilemap_layers: dict[str, list[list[int]]]
    composite_map_path: str
    manual_path: str
    manual_markdown: str
    assets: dict[str, SpriteResponse] = {}
