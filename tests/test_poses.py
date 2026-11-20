"""Poses OpenPose e o ramo ControlNet do ComfyProvider (sem rede)."""
from __future__ import annotations

import numpy as np
import pytest

from openpixel_core.comfy_provider import ComfyProvider
from openpixel_core.poses import PRESETS, preset_image


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_image_has_skeleton(name):
    img = preset_image(name, 256)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8 and img.any() and (img == 0).all(axis=2).mean() > 0.9


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        preset_image("voando")


def test_workflow_with_pose_routes_conditioning_through_controlnet():
    wf = ComfyProvider(loras="none").build_workflow("knight", 1, pose_image="p.png")
    assert wf["pose_in"]["inputs"]["image"] == "p.png"
    assert wf["cn"]["inputs"]["positive"] == ["3", 0] and wf["cn"]["inputs"]["negative"] == ["4", 0]
    assert wf["6"]["inputs"]["positive"] == ["cn", 0] and wf["6"]["inputs"]["negative"] == ["cn", 1]


def test_workflow_without_pose_has_no_controlnet():
    wf = ComfyProvider(loras="none").build_workflow("knight", 1)
    assert "cn" not in wf and wf["6"]["inputs"]["positive"] == ["3", 0]


def test_workflow_with_reference_uses_ip_adapter_on_model():
    wf = ComfyProvider(loras="a.safetensors").build_workflow("knight", 1, ref_image="r.png", ref_weight=0.5)
    assert wf["ipa_load"]["inputs"]["model"] == ["lora0", 0]
    assert wf["ipa"]["inputs"]["weight"] == 0.5 and wf["6"]["inputs"]["model"] == ["ipa", 0]


def test_skeleton_feet_touch_ground_and_rejects_unknown_params():
    from openpixel_core.poses import skeleton

    pts = skeleton()
    assert len(pts) == 18 and max(pts[10][1], pts[13][1]) == pytest.approx(0.89)
    with pytest.raises(ValueError):
        skeleton(voar=1.0)


def test_interpolate_is_cyclic_and_validates_keys():
    from openpixel_core.poses import interpolate

    poses = interpolate([(0.0, {"chest": 0.0}), (0.5, {"chest": 0.02})], frames=8)
    necks = [p[1][1] for p in poses]
    assert len(poses) == 8 and necks[4] < necks[0]          # no meio o peito esta mais alto (y menor)
    assert abs(necks[7] - necks[0]) < abs(necks[4] - necks[0])  # volta ao inicio: o ultimo quadro se aproxima do primeiro
    with pytest.raises(ValueError):
        interpolate([(1.5, {})], frames=4)


def test_idle_cycle_moves_chest_a_little_and_keeps_feet_planted():
    from openpixel_core.poses import idle_cycle

    poses = idle_cycle(8)
    chest = [p[1][1] for p in poses]
    assert 0.01 < max(chest) - min(chest) < 0.04
    assert all(abs(max(p[10][1], p[13][1]) - 0.89) < 1e-6 for p in poses)
