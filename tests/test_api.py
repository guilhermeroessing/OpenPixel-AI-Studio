"""Testes da API FastAPI (backend-core) via TestClient."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from openpixel_core import api

DIRECTIONS = {"S", "SE", "E", "NE", "N", "NW", "W", "SW"}


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "out"
    monkeypatch.setattr(api, "OUT_ROOT", root)
    return root


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_generate_returns_sprite(client):
    r = client.post("/generate", json={"prompt": "grass tree", "size": 32, "seed": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 32 and body["height"] == 32
    assert body["seed"] == 7
    assert body["provider"] == "smart_mock"
    assert isinstance(body["image_b64"], str) and body["image_b64"]
    assert 1 <= len(body["palette"]) <= 16


def test_generate_rejects_invalid_size(client):
    r = client.post("/generate", json={"prompt": "grass", "size": 8})
    assert r.status_code == 422


def test_generate_rejects_empty_prompt(client):
    r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422


def test_generate8_returns_all_directions(client):
    r = client.post("/generate8", json={"prompt": "knight", "size": 32, "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == DIRECTIONS
    palettes = {tuple(v["palette"]) for v in body.values()}
    assert len(palettes) == 1


def test_map_with_explicit_out_dir(client, tmp_path):
    out = tmp_path / "map_explicit"
    r = client.post("/map", json={"prompt": "grass lake tree", "out_dir": str(out)})
    assert r.status_code == 200
    body = r.json()
    assert body["manual_markdown"].startswith("# MAP_ASSEMBLY_MANUAL")
    assert (out / "MAP_ASSEMBLY_MANUAL.md").is_file()
    assert (out / "composite_map.png").is_file()
    assert (out / "map.json").is_file()
    assert len(body["tilemap_layers"]["ground"]) == 16
    assert len(body["tilemap_layers"]["ground"][0]) == 16


def test_map_default_out_dir_uses_output_root(client, out_root):
    r = client.post("/map", json={"prompt": "desert cactus"})
    assert r.status_code == 200
    created = list(out_root.glob("map_*"))
    assert len(created) == 1
    assert (created[0] / "MAP_ASSEMBLY_MANUAL.md").is_file()


def test_map_rejects_missing_prompt(client):
    r = client.post("/map", json={})
    assert r.status_code == 422


def test_vram_falls_back_without_nvidia_smi(client, monkeypatch):
    monkeypatch.setattr(api.shutil, "which", lambda _name: None)
    r = client.get("/vram")
    assert r.status_code == 200
    assert r.json() == {"mode": "cpu_mock", "used_mb": 0, "total_mb": 0}


def test_vram_falls_back_when_nvidia_smi_fails(client, monkeypatch):
    monkeypatch.setattr(api.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def boom(*_a, **_k):
        raise RuntimeError("driver ausente")

    monkeypatch.setattr(api.subprocess, "run", boom)
    r = client.get("/vram")
    assert r.status_code == 200
    assert r.json()["mode"] == "cpu_mock"


def test_dataset_empty_when_db_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "inexistente.db")
    r = client.get("/dataset")
    assert r.status_code == 200
    assert r.json() == []


def _make_catalog(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, source TEXT, path TEXT, caption_path TEXT, "
                "kind TEXT, width INTEGER, height INTEGER, scale INTEGER, palette_json TEXT, approved INTEGER)")
    rows = [
        (1, "scraper", "a.png", "a.txt", "tile", 32, 32, 1, '["#000000", "#ffffff"]', 1),
        (2, "scraper", "b.png", None, "character", 64, 64, 2, "not-json", 0),
        (3, "scraper", "c.png", None, "asset", 16, 16, 1, None, 0),
    ]
    con.executemany("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_dataset_returns_rows_with_parsed_palette(client, monkeypatch, tmp_path):
    db = tmp_path / "catalog.db"
    _make_catalog(db)
    monkeypatch.setattr(api, "DB_PATH", db)
    r = client.get("/dataset")
    assert r.status_code == 200
    rows = {row["id"]: row for row in r.json()}
    assert set(rows) == {1, 2, 3}
    assert rows[1]["palette"] == ["#000000", "#ffffff"]
    assert "palette_json" not in rows[1]
    assert rows[2]["palette"] == []  # JSON invalido -> lista vazia
    assert rows[3]["palette"] == []  # NULL -> lista vazia


def test_dataset_respects_limit(client, monkeypatch, tmp_path):
    db = tmp_path / "catalog.db"
    _make_catalog(db)
    monkeypatch.setattr(api, "DB_PATH", db)
    r = client.get("/dataset", params={"limit": 2})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_dataset_returns_empty_on_sqlite_error(client, monkeypatch, tmp_path):
    db = tmp_path / "corrompido.db"
    db.write_bytes(b"isto nao e um banco sqlite" * 10)
    monkeypatch.setattr(api, "DB_PATH", db)
    r = client.get("/dataset")
    assert r.status_code == 200
    assert r.json() == []
