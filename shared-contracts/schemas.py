"""Contratos compartilhados (Pydantic) entre backend, gateway MCP e frontend."""
from typing import Literal
from pydantic import BaseModel, Field

AssetKind = Literal["character", "asset", "tileset", "map"]


class GenParams(BaseModel):
    """Forca de cada fator da geracao. None = usa o padrao do provider (ver GET /params)."""
    lora_pixel_art_xl: float | None = Field(default=None, ge=0, le=2)
    lora_redmond: float | None = Field(default=None, ge=0, le=2)
    cfg: float | None = Field(default=None, ge=1, le=20)
    steps: int | None = Field(default=None, ge=5, le=60)
    pose_strength: float | None = Field(default=None, ge=0, le=2)
    ref_weight: float | None = Field(default=None, ge=0, le=2)
    shoulder_spread: float | None = Field(default=None, ge=0, le=0.1)
    flight_height: float | None = Field(default=None, ge=0, le=0.1)
    lean: float | None = Field(default=None, ge=0, le=0.7)
    bg_tolerance: int | None = Field(default=None, ge=10, le=150)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    size: int = Field(default=32, ge=16, le=128)
    kind: AssetKind = "asset"
    palette_size: int = Field(default=16, ge=2, le=64)
    seamless: bool = False
    symmetric: bool = False
    dither: bool = False
    seed: int | None = None
    params: GenParams | None = None


Motion = Literal["run", "idle", "punch", "sword"]  # movimentos implementados (ver openpixel_core.animation.MOTIONS)


class AnimateRequest(GenerateRequest):
    frames: int | None = Field(default=None, ge=2, le=16)  # None = o padrao do movimento (corrida 8, idle 4)
    motion: Motion = "run"


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


class AnimationResponse(BaseModel):
    frames: list[SpriteResponse]
    seed: int
    fps: int = 10
    provider: str = "smart_mock"


class CompleteMapResponse(BaseModel):
    master_palette: list[str]
    asset_manifest: list[MapAsset]
    tilemap_layers: dict[str, list[list[int]]]
    composite_map_path: str
    manual_path: str
    manual_markdown: str
    assets: dict[str, SpriteResponse] = {}
