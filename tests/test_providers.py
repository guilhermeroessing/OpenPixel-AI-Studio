"""Selecao de provider por DEVICE_MODE, sem precisar de GPU."""
from __future__ import annotations

import sys
import types

from openpixel_core import providers
from openpixel_core.mock_provider import SmartMockProvider
from openpixel_core.sdxl_provider import SDXLProvider


def test_cpu_mock_default(monkeypatch):
    monkeypatch.setenv("DEVICE_MODE", "cpu_mock")
    assert isinstance(providers.get_provider(), SmartMockProvider)


def test_cuda_without_cuda_falls_back(monkeypatch):
    monkeypatch.setenv("DEVICE_MODE", "cuda")
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    assert isinstance(providers.get_provider(), SmartMockProvider)


def test_cuda_without_lora_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("DEVICE_MODE", "cuda")
    monkeypatch.setenv("SDXL_LORA_PATH", str(tmp_path / "nao_existe.safetensors"))
    monkeypatch.delenv("OPENPIXEL_ALLOW_BASE_SDXL", raising=False)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setattr(providers, "_dotenv_value", lambda key: None)
    assert isinstance(providers.get_provider(), SmartMockProvider)


def test_cuda_with_lora_returns_sdxl(monkeypatch, tmp_path):
    lora = tmp_path / "x.safetensors"
    lora.write_bytes(b"")
    monkeypatch.setenv("DEVICE_MODE", "cuda")
    monkeypatch.setenv("SDXL_LORA_PATH", str(lora))
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    p = providers.get_provider()
    assert isinstance(p, SDXLProvider) and p.name == "sdxl_lora" and p.lora_path == lora


def test_sdxl_provider_is_lazy_and_builds_prompt():
    p = SDXLProvider()
    assert p._pipe is None  # nada de torch/diffusers carregado no construtor
    assert SDXLProvider.build_prompt(" fire ").startswith("openpixel, pixel art, fire")
