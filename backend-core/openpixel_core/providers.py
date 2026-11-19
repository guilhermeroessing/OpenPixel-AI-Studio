"""Escolha do provider de imagem conforme DEVICE_MODE (cpu_mock | cuda)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .mock_provider import SmartMockProvider

log = logging.getLogger("openpixel.providers")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_DIR = ROOT / "training-local" / "output" / "sdxl-lora"


def _dotenv_value(key: str) -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    return None


def _setting(key: str, default: str | None = None) -> str | None:
    """Variavel de ambiente tem prioridade sobre o .env."""
    return os.environ.get(key) or _dotenv_value(key) or default


def _find_lora() -> Path | None:
    explicit = _setting("SDXL_LORA_PATH")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    if DEFAULT_LORA_DIR.is_dir():
        files = sorted(DEFAULT_LORA_DIR.glob("*.safetensors"))
        if files:
            return files[-1]
    return None


def get_provider():
    mode = (_setting("DEVICE_MODE", "cpu_mock") or "cpu_mock").lower()
    if mode == "comfy":
        # ComfyUI local; se nao estiver aberto no momento da geracao, o erro explica. Nao cai no mock
        # de proposito: o usuario escolheu o ComfyUI e um mock silencioso esconderia o problema.
        from .comfy_provider import ComfyProvider  # noqa: PLC0415

        return ComfyProvider(url=_setting("COMFY_URL"))
    if mode != "cuda":
        return SmartMockProvider()

    # Modo cuda: so usa o SDXL se for realmente utilizavel; senao cai no mock com aviso,
    # para o backend continuar subindo (ex.: antes de existir uma LoRA treinada).
    try:
        import diffusers  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponivel")
    except Exception as exc:  # noqa: BLE001
        log.warning("DEVICE_MODE=cuda mas o SDXL nao esta utilizavel (%s); usando SmartMockProvider.", exc)
        return SmartMockProvider()

    lora = _find_lora()
    if lora is None and _setting("OPENPIXEL_ALLOW_BASE_SDXL") != "1":
        log.warning(
            "DEVICE_MODE=cuda mas nao ha LoRA (SDXL_LORA_PATH ou %s); usando SmartMockProvider. "
            "Defina OPENPIXEL_ALLOW_BASE_SDXL=1 para usar o SDXL base sem LoRA.",
            DEFAULT_LORA_DIR,
        )
        return SmartMockProvider()

    from .sdxl_provider import SDXLProvider  # noqa: PLC0415

    return SDXLProvider(lora_path=lora)
