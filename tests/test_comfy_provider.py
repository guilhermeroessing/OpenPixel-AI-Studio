"""ComfyProvider com servidor ComfyUI simulado (sem rede, sem GPU)."""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from openpixel_core import providers
from openpixel_core.comfy_provider import ComfyProvider, parse_loras


def test_parse_loras():
    assert parse_loras("a.safetensors:0.7,b.safetensors") == [("a.safetensors", 0.7), ("b.safetensors", 1.0)]
    assert parse_loras("none") == [] and parse_loras("") == []


def test_workflow_stacks_loras_and_applies_prompt():
    p = ComfyProvider(loras="a.safetensors:0.5,b.safetensors:0.8", trigger="Pixel Art", avoid="(shield:1.8)")
    wf = p.build_workflow("goblin", 7)
    assert wf["lora0"]["inputs"]["model"] == ["1", 0] and wf["lora0"]["inputs"]["strength_model"] == 0.5
    assert wf["lora1"]["inputs"]["model"] == ["lora0", 0] and wf["lora1"]["inputs"]["lora_name"] == "b.safetensors"
    assert wf["6"]["inputs"]["model"] == ["lora1", 0] and wf["6"]["inputs"]["seed"] == 7
    assert wf["3"]["inputs"]["clip"] == ["lora1", 1]
    assert wf["3"]["inputs"]["text"].startswith("Pixel Art, goblin")
    assert wf["4"]["inputs"]["text"].endswith("(shield:1.8)")


def test_workflow_without_loras_uses_checkpoint_directly():
    wf = ComfyProvider(loras="none").build_workflow("tree", 1)
    assert wf["6"]["inputs"]["model"] == ["1", 0] and wf["3"]["inputs"]["clip"] == ["1", 1]


def test_generate_round_trip_pixelates(monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 30)).save(buf, format="PNG")
    png = buf.getvalue()
    p = ComfyProvider(url="http://fake", native_size=64, grid=8, colors=4)
    sent = {}

    def fake_http(path, data=None, raw=False, timeout=60):
        if path == "/prompt":
            sent["wf"] = json.loads(data)["prompt"]
            return {"prompt_id": "abc"}
        if path.startswith("/history/"):
            return {"abc": {"status": {"status_str": "success"},
                            "outputs": {"8": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}}}}
        if path.startswith("/view"):
            return png
        raise AssertionError(path)

    monkeypatch.setattr(p, "_http", fake_http)
    img, seed = p.generate("grass tileset", 32, 5)
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8 and seed == 5
    assert "lora0" in sent["wf"]


def test_providers_selects_comfy(monkeypatch):
    monkeypatch.setenv("DEVICE_MODE", "comfy")
    assert providers.get_provider().name == "comfyui_sdxl_pixel"
