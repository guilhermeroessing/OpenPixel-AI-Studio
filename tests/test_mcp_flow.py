"""Fluxo MCP: tools do ai-gateway-mcp via FastMCP Client em memoria."""
from __future__ import annotations

import asyncio
import json
import zipfile

import pytest

from fastmcp import Client
from fastmcp.exceptions import ToolError

from server import mcp

EXPECTED_TOOLS = {"generate_pixel_art", "generate_8_directions", "animate_asset",
                  "generate_complete_map", "export_engine_bundle"}


def _call(name: str, args: dict):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, args)

    return asyncio.run(run())


def _call_error(name: str, args: dict):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, args)

    with pytest.raises(ToolError) as exc:
        asyncio.run(run())
    return str(exc.value)


@pytest.fixture(scope="module")
def map_dir(tmp_path_factory):
    """Gera um mapa uma vez para o modulo (usado nos testes de export)."""
    out = tmp_path_factory.mktemp("mcp_map") / "map"
    result = _call("generate_complete_map", {"prompt": "grass lake tree rock", "out_dir": str(out)})
    assert not result.is_error
    return out


def test_server_exposes_expected_tools():
    async def run():
        async with Client(mcp) as client:
            return {t.name for t in await client.list_tools()}

    assert asyncio.run(run()) >= EXPECTED_TOOLS


def test_generate_complete_map_returns_manual_and_files(tmp_path):
    out = tmp_path / "map"
    result = _call("generate_complete_map", {"prompt": "grass lake tree", "out_dir": str(out)})
    assert not result.is_error

    data = result.structured_content
    assert data["manual_markdown"].strip(), "manual_markdown nao pode ser vazio"
    assert "MAP_ASSEMBLY_MANUAL" in data["manual_markdown"]
    assert data["master_palette"]
    assert data["asset_manifest"]

    assert (out / "MAP_ASSEMBLY_MANUAL.md").is_file()
    assert (out / "composite_map.png").is_file()
    assert (out / "map.json").is_file()
    assert (out / "assets").is_dir()
    assert data["manual_path"] == str(out / "MAP_ASSEMBLY_MANUAL.md")
    assert data["composite_map_path"] == str(out / "composite_map.png")


def test_generate_complete_map_manual_text_matches_file(tmp_path):
    out = tmp_path / "map"
    data = _call("generate_complete_map", {"prompt": "desert cactus", "out_dir": str(out)}).structured_content
    assert data["manual_markdown"] == (out / "MAP_ASSEMBLY_MANUAL.md").read_text(encoding="utf-8")


def test_generate_complete_map_assets_are_written(tmp_path):
    out = tmp_path / "map"
    data = _call("generate_complete_map", {"prompt": "snow stone house", "out_dir": str(out)}).structured_content
    for asset in data["asset_manifest"]:
        assert (out / "assets" / f"{asset['name']}.png").is_file()


def test_export_godot_bundle_is_zip(map_dir):
    result = _call("export_engine_bundle", {"map_dir": str(map_dir), "engine": "godot"})
    assert not result.is_error
    bundle = result.structured_content["bundle_path"]
    assert result.structured_content["engine"] == "godot"
    assert bundle.endswith("_godot_bundle.zip")
    with zipfile.ZipFile(bundle) as z:
        names = set(z.namelist())
        assert "godot_map.json" in names
        assert "MAP_ASSEMBLY_MANUAL.md" in names
        assert "composite_map.png" in names
        assert any(n.startswith("assets/") and n.endswith(".png") for n in names)
        doc = json.loads(z.read("godot_map.json"))
        assert doc["format"] == "openpixel-godot-4"


def test_export_unity_bundle_is_zip(map_dir):
    result = _call("export_engine_bundle", {"map_dir": str(map_dir), "engine": "unity"})
    assert not result.is_error
    bundle = result.structured_content["bundle_path"]
    with zipfile.ZipFile(bundle) as z:
        assert "unity_map.json" in z.namelist()
        doc = json.loads(z.read("unity_map.json"))
        assert doc["format"] == "openpixel-unity"


def test_export_rejects_unknown_engine(map_dir):
    message = _call_error("export_engine_bundle", {"map_dir": str(map_dir), "engine": "tiled"})
    assert "godot" in message and "unity" in message


def test_generate_pixel_art_tool(tmp_path):
    data = _call("generate_pixel_art", {"prompt": "grass", "size": 32, "seed": 5}).structured_content
    assert data["width"] == 32 and data["height"] == 32
    assert data["seed"] == 5


def test_generate_8_directions_tool():
    data = _call("generate_8_directions", {"prompt": "knight", "size": 32, "seed": 5}).structured_content
    assert set(data) == {"S", "SE", "E", "NE", "N", "NW", "W", "SW"}


def test_animate_asset_tool_returns_frames():
    result = _call("animate_asset", {"prompt": "slime", "frames": 4, "size": 32, "seed": 5})
    frames = result.data
    assert len(frames) == 4
    assert len({tuple(f["palette"]) for f in frames}) == 1
