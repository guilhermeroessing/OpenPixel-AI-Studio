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


def test_anchor_fixed_keeps_relative_position_while_centroid_recenters():
    a, b = _scene(0), np.roll(_scene(0), 40, axis=1)  # mesmo personagem 40 px (10 na grade) mais a direita
    left = lambda f: int(np.nonzero(f[..., 3].any(axis=0))[0].min())
    c = animation.process_frames([a, b], grid=64, colors=8, anchor="centroid")
    f = animation.process_frames([a, b], grid=64, colors=8, anchor="fixed")
    assert left(c[0]) == left(c[1])               # recentralizado: iguais
    assert left(f[1]) - left(f[0]) == 10          # local fixo: o deslocamento da pose e preservado
    import pytest
    with pytest.raises(ValueError):
        animation.process_frames([a], anchor="solto")


def test_avoid_for_adds_motion_specific_terms():
    assert animation.avoid_for(animation.MOTIONS["run"]) == animation.RUN_AVOID
    punch = animation.avoid_for(animation.MOTIONS["punch"])
    assert punch.startswith(animation.RUN_AVOID) and "boxing gloves" in punch and "smoke" in punch


def test_prompt_weight_wraps_character_text_only_when_requested():
    class Fake:
        native_size = 64
        prompts: list = []

        def generate(self, text, size, seed, pose=None, reference=None):
            self.prompts.append(text)
            return np.zeros((size, size, 3), np.uint8), seed

    pose = [(0.5, 0.5)] * 18
    f = Fake()
    animation.generate_motion_raws(f, "girl", 64, 1, [pose], "punching", False)
    animation.generate_motion_raws(f, "girl", 64, 1, [pose], "punching", False, prompt_weight=1.3)
    assert f.prompts == ["girl, punching", "(girl:1.3), punching"]


def test_sword_motion_forbids_weapons_in_the_prompt():
    sword = animation.avoid_for(animation.MOTIONS["sword"])
    assert "(sword:1.6)" in sword and sword.startswith(animation.RUN_AVOID)


def test_remove_bg_keeps_white_torso_that_matches_the_background_when_protected():
    g = 64
    a = np.full((g, g, 3), 235, np.uint8)                    # fundo claro
    a[10:20, 28:36] = (200, 30, 30)                          # cabeca/pescoco
    a[48:60, 24:40] = (30, 30, 160)                          # shorts e pernas
    a[20:48, 28:36] = 236                                    # regata branca ~ cor do fundo, ligada a borda
    a[20:48, 20:28] = 235
    pose = [(0.5, 0.2)] * 18
    pose[1] = (0.5, 0.3)
    pose[8], pose[11] = (0.5, 0.7), (0.5, 0.7)
    sem = animation._remove_bg(a, 45)
    com = animation._remove_bg(a, 45, protect=animation._torso_protect(pose, g))
    assert not sem[34, 32] and com[34, 32]
    assert not com[34, 5]                                    # o fundo longe do tronco continua sendo fundo


def test_remove_bg_leak_guard_blocks_one_pixel_gaps_into_white_clothing():
    g = 48
    a = np.full((g, g, 3), 235, np.uint8)
    a[8:40, 14:34] = (40, 40, 40)                            # corpo
    a[16:30, 18:30] = 236                                    # roupa branca (cor do fundo) dentro do corpo
    a[22, 14:19] = 236                                       # fresta de 1 px ligando a roupa ao fundo
    sem = animation._remove_bg(a, 45)
    com = animation._remove_bg(a, 45, leak_guard=True)
    assert not sem[22, 24] and com[22, 24]
    assert not com[2, 2]                                     # o fundo aberto continua sendo fundo


def _pose(dx: float = 0.0):
    return [[0.5 + dx, 0.1 + 0.05 * (i % 7)] for i in range(18)]


def test_motions_endpoint_lists_presets_with_18_point_poses():
    r = client.get("/motions").json()
    assert {"run", "idle", "punch", "sword"} <= set(r)
    for m in r.values():
        assert len(m["poses"]) == m["frames"] and all(len(p) == 18 for p in m["poses"])


def test_animate_request_validates_custom_poses():
    import pytest
    from pydantic import ValidationError
    from schemas import AnimateRequest

    ok = AnimateRequest(prompt="girl", motion="custom", poses=[_pose(), _pose(0.1)])
    assert len(ok.poses) == 2
    with pytest.raises(ValidationError):
        AnimateRequest(prompt="girl", motion="custom")                          # custom exige poses
    with pytest.raises(ValidationError):
        AnimateRequest(prompt="girl", motion="custom", poses=[_pose()[:17]])    # 17 pontos
    with pytest.raises(ValidationError):
        AnimateRequest(prompt="girl", motion="custom", poses=[[[5.0, 5.0]] * 18])  # fora da imagem
    many = AnimateRequest(prompt="girl", motion="custom", poses=[_pose()] * 100)  # sem teto baixo de frames
    assert len(many.poses) == 100


def test_custom_motion_forbids_weapons_unless_the_user_allows_objects():
    poses = [_pose(), _pose(0.1)]
    safe = animation.custom_motion(poses, "waving hello", extra_avoid="hat")
    assert animation.avoid_for(safe).startswith(animation.RUN_AVOID) and animation.avoid_for(safe).endswith("hat")
    assert safe["suffix"].startswith("waving hello") and safe["frames"] == 2 and safe["anchor"] == "fixed"
    free = animation.custom_motion(poses, "", allow_objects=True)
    assert "(sword:" not in animation.avoid_for(free) and "cropped" in animation.avoid_for(free)
    assert free["poses"](2, {}) is poses


def test_animate_custom_poses_reach_the_generator_with_one_pose_per_frame(monkeypatch):
    from openpixel_core import service

    seen = {}

    class Fake:
        native_size = 64
        ref_weight = 0.5
        avoid = ""

        def with_overrides(self, **kw):
            seen["avoid"] = kw.get("avoid")
            return self

    def fake_raws(prov, prompt, native, seed, poses, suffix, use_ref, progress=None, prompt_weight=1.0):
        seen.update(n=len(poses), suffix=suffix)
        return [np.full((64, 64, 3), 150, np.uint8) for _ in poses]

    monkeypatch.setattr(service, "_provider_for", lambda params: Fake())
    monkeypatch.setattr(animation, "generate_motion_raws", fake_raws)
    from schemas import AnimateRequest

    req = AnimateRequest(prompt="girl", size=32, motion="custom", poses=[_pose(), _pose(0.05), _pose(0.1)],
                         motion_text="waving", fps=5, seed=3)
    res = service.animate_run(req)
    assert seen["n"] == 3 and seen["suffix"].startswith("waving") and res.fps == 5 and len(res.frames) == 3


def test_motions_presets_carry_the_description_and_avoid_terms_for_the_pose_editor():
    r = client.get("/motions").json()
    assert "fist extended" in r["punch"]["text"] and "full body" not in r["punch"]["text"]
    assert "gloves" in r["punch"]["avoid"] and r["idle"]["avoid"] == ""


def test_character_reference_is_neutral_and_independent_of_the_motion_text():
    class Fake:
        native_size = 64

        def __init__(self):
            self.calls = []

        def generate(self, text, size, seed, pose=None, reference=None):
            self.calls.append((text, reference))
            return np.zeros((size, size, 3), np.uint8), seed

    pose = [(0.5, 0.5)] * 18
    f = Fake()
    animation.generate_motion_raws(f, "girl", 64, 1, [pose, pose], "throwing a punch", True)
    assert f.calls[0] == (f"girl, {animation.REF_SUFFIX}", None)                 # referencia: sem "soco"
    assert all("throwing a punch" in t and r is not None for t, r in f.calls[1:])  # frames: movimento + referencia
