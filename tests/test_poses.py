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
