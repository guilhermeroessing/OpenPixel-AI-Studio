"""Animacao: parametros editaveis, tarefa em segundo plano e pos-processamento dos frames (sem GPU/ComfyUI)."""
from __future__ import annotations

import base64
import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from openpixel_core import animation
from openpixel_core.api import app
from openpixel_core.comfy_provider import ComfyProvider

client = TestClient(app)


def _scene(shift_y: int = 0, size: int = 256) -> np.ndarray:
    """Fundo liso cinza, 'personagem' vermelho com um bloco cinza-escuro de sombra (cor parecida com o fundo)."""
    img = np.full((size, size, 3), (150, 150, 150), np.uint8)
    img[60 + shift_y:200 + shift_y, 100:150] = (200, 30, 30)
    return img


def test_params_lists_specs_and_defaults():
    r = client.get("/params").json()
    keys = {s["key"] for s in r["specs"]}
    assert {"lora_pixel_art_xl", "pose_strength", "ref_weight", "cfg", "bg_tolerance"} <= keys
    assert keys == set(r["defaults"]) and all(s["min"] <= r["defaults"][s["key"]] <= s["max"] for s in r["specs"])


def test_params_validation_rejects_out_of_range():
    body = {"prompt": "x", "size": 32, "frames": 4, "params": {"cfg": 99}}
    assert client.post("/animate", json=body).status_code == 422


def test_animate_job_round_trip_with_mock():
    jid = client.post("/animate", json={"prompt": "runner", "size": 32, "frames": 4}).json()["job_id"]
    for _ in range(100):
        j = client.get(f"/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    frames = j["result"]["frames"]
    assert len(frames) == 4 and all(f["width"] == 32 for f in frames)
    Image.open(io.BytesIO(base64.b64decode(frames[0]["image_b64"]))).verify()


def test_unknown_job_is_404():
    assert client.get("/jobs/nope").status_code == 404


def test_with_overrides_changes_copy_only():
    p = ComfyProvider(loras="pixel-art-xl-v1.1.safetensors:0.7,PixelArtRedmond-Lite64.safetensors:0.7", cfg=6.5)
    c = p.with_overrides(cfg=9.0, ref_weight=0.2, lora_strengths={"redmond": 0.3})
    assert c.cfg == 9.0 and c.ref_weight == 0.2 and dict(c.loras)["PixelArtRedmond-Lite64.safetensors"] == 0.3
    assert dict(c.loras)["pixel-art-xl-v1.1.safetensors"] == 0.7
    assert p.cfg == 6.5 and p.loras[1][1] == 0.7  # original intacto


def test_process_frames_transparent_bg_shared_size_and_vertical_motion_kept():
    out = animation.process_frames([_scene(0), _scene(-20)], grid=64, colors=8)
    assert all(o.shape == (64, 64, 4) for o in out)
    assert (out[0][0, 0, 3] == 0) and (out[0][..., 3] == 255).any()  # fundo transparente, personagem opaco
    top = [int(np.nonzero(o[..., 3].any(axis=1))[0].min()) for o in out]
    assert top[1] < top[0]  # o frame deslocado para cima continua mais alto: o voo nao foi apagado


def test_process_frames_removes_flat_ground_shadow():
    img = _scene(0)
    img[200:216, 60:190] = (128, 128, 128)  # faixa larga e GROSSA (4 linhas na grade) abaixo dos pes (sombra)
    out = animation.process_frames([img], grid=64, colors=8, bg_tol=45)[0]
    ys = np.nonzero(out[..., 3].any(axis=1))[0]
    assert ys.max() - ys.min() < 37  # sem a faixa o personagem ocupa ~ 35 linhas de 64; com ela, 39


def test_animate_accepts_idle_and_rejects_unknown_motion():
    ok = client.post("/animate", json={"prompt": "x", "size": 32, "frames": 4, "motion": "idle"})
    assert ok.status_code == 200
    assert client.post("/animate", json={"prompt": "x", "size": 32, "motion": "voar"}).status_code == 422


def test_process_frames_keeps_wide_shoes_but_drops_wider_shadow():
    img = np.full((256, 256, 3), (150, 150, 150), np.uint8)
    img[40:200, 120:136] = (200, 30, 30)          # corpo/perna estreita
    img[200:216, 108:148] = (240, 240, 240)       # tenis: 40 px de largura em 256 (~16%, abaixo de 20%)
    out = animation.process_frames([img], grid=64, colors=8, bg_tol=45)[0]
    ys = np.nonzero(out[..., 3].any(axis=1))[0]
    assert ys.max() - ys.min() >= 39              # os 4 ultimos pixels do tenis continuam la
