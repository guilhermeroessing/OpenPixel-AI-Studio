"""Contratos compartilhados (Pydantic) entre backend, gateway MCP e frontend."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator

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


Motion = Literal["run", "idle", "punch", "sword", "custom"]  # prontos (openpixel_core.animation.MOTIONS) ou "custom"
MAX_FRAMES = 240          # sem teto pratico: cada computador gera no seu tempo
POSE_POINTS = 18          # esqueleto COCO-18 (ver openpixel_core.poses)


class AnimateRequest(GenerateRequest):
    frames: int | None = Field(default=None, ge=2, le=MAX_FRAMES)  # None = o padrao do movimento (corrida 8, idle 4)
    motion: Motion = "run"
    # movimento personalizado: uma pose (18 pontos x, y em 0..1) por frame, desenhada pelo usuario no editor de poses
    poses: list[list[tuple[float, float]]] | None = None
    motion_text: str | None = Field(default=None, max_length=300)    # descricao do movimento (sem nomear armas)
    allow_objects: bool = False                                      # True = nao proibe armas/objetos (por conta do usuario)
    extra_avoid: str | None = Field(default=None, max_length=300)    # termos negativos extras
    fps: int | None = Field(default=None, ge=1, le=60)
    anchor: Literal["fixed", "centroid"] | None = None

    @model_validator(mode="after")
    def _check_custom_poses(self):
        if self.poses is None:
            if self.motion == "custom":
                raise ValueError("o movimento 'custom' precisa de 'poses'")
            return self
        if not 1 <= len(self.poses) <= MAX_FRAMES:
            raise ValueError(f"numero de poses deve ficar entre 1 e {MAX_FRAMES}")
        for pose in self.poses:
            if len(pose) != POSE_POINTS:
                raise ValueError(f"cada pose precisa de {POSE_POINTS} pontos")
            if any(not (-0.25 <= x <= 1.25 and -0.25 <= y <= 1.25) for x, y in pose):
                raise ValueError("coordenadas das poses devem ficar proximas de 0..1")
        return self


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
