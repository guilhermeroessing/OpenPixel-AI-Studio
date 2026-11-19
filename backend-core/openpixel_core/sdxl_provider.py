"""SDXLProvider: gera a imagem de alta resolucao com SDXL (+ LoRA do OpenPixel) na GPU.

Mesma interface do SmartMockProvider: generate(prompt, size, seed) -> (RGB uint8 HxWx3, seed).
torch/diffusers so sao importados quando o pipeline e carregado (primeira geracao), entao
importar este modulo e barato e funciona em maquinas sem GPU.
"""
from __future__ import annotations

import os
import random
import threading
from pathlib import Path

import numpy as np

TRIGGER = "openpixel"
DEFAULT_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
NEGATIVE = "blurry, anti-aliased, smooth gradient, photo, realistic, 3d render, jpeg artifacts, text, watermark"


class SDXLProvider:
    name = "sdxl_lora"

    def __init__(
        self,
        lora_path: str | Path | None = None,
        base_model: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        native_size: int | None = None,
    ):
        self.lora_path = Path(lora_path) if lora_path else None
        self.base_model = base_model or os.environ.get("SDXL_BASE_MODEL", DEFAULT_BASE)
        self.steps = steps or int(os.environ.get("SDXL_STEPS", "28"))
        self.guidance = guidance or float(os.environ.get("SDXL_GUIDANCE", "6.5"))
        # 8 GB de VRAM: 1024 e o tamanho nativo do SDXL; 768 e o fallback via SDXL_SIZE
        self.native_size = native_size or int(os.environ.get("SDXL_SIZE", "1024"))
        self._pipe = None
        self._lock = threading.Lock()

    # ---- pipeline preguicoso -------------------------------------------------
    def _load(self):
        import torch  # noqa: PLC0415 (import preguicoso de proposito)
        from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("SDXLProvider exige CUDA (torch.cuda.is_available() == False)")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        if self.lora_path is not None:
            if self.lora_path.is_dir():
                pipe.load_lora_weights(str(self.lora_path))
            else:
                pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()  # cabe em 8 GB: so o modulo ativo fica na GPU
        return pipe

    def _pipeline(self):
        with self._lock:
            if self._pipe is None:
                self._pipe = self._load()
            return self._pipe

    # ---- interface do provider ----------------------------------------------
    @staticmethod
    def build_prompt(prompt: str) -> str:
        return f"{TRIGGER}, pixel art, {prompt.strip()}, hard pixel edges, limited palette"

    def generate(self, prompt: str, size: int = 1024, seed: int | None = None):
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        pipe = self._pipeline()
        gen = torch.Generator("cpu").manual_seed(seed)
        out = pipe(
            prompt=self.build_prompt(prompt),
            negative_prompt=NEGATIVE,
            width=self.native_size,
            height=self.native_size,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=gen,
        ).images[0]
        if out.size != (size, size):
            out = out.resize((size, size), Image.LANCZOS)
        return np.ascontiguousarray(np.asarray(out.convert("RGB"), dtype=np.uint8)), seed
