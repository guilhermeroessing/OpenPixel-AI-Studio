"""API FastAPI do backend-core."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import service
from schemas import GenerateRequest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("DATASET_DB", ROOT / "scraper-dataset" / "dataset_catalog.db"))
OUT_ROOT = Path(os.environ.get("OPENPIXEL_OUTPUT", ROOT / "backend-core" / "output"))

app = FastAPI(title="OpenPixel Core")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class MapRequest(BaseModel):
    prompt: str
    out_dir: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    return service.generate_pixel_art(req)


@app.post("/generate8")
def generate8(req: GenerateRequest):
    return service.generate_8_directions(req)


@app.post("/map")
def map_(req: MapRequest):
    out = req.out_dir or str(OUT_ROOT / ("map_" + hashlib.sha1(req.prompt.encode()).hexdigest()[:10]))
    return service.generate_complete_map(req.prompt, out)


@app.get("/vram")
def vram():
    fallback = {"mode": "cpu_mock", "used_mb": 0, "total_mb": 0}
    if not shutil.which("nvidia-smi"):
        return fallback
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5, check=True)
        used, total = [int(float(x)) for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"mode": "gpu", "used_mb": used, "total_mb": total}
    except Exception:
        return fallback


@app.get("/dataset")
def dataset(limit: int = 200):
    if not DB_PATH.exists():
        return []
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT id,source,path,caption_path,kind,width,height,scale,palette_json,approved "
                           "FROM assets LIMIT ?", (limit,)).fetchall()
        con.close()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["palette"] = json.loads(d.pop("palette_json") or "[]")
        except (TypeError, ValueError):
            d["palette"] = []
        out.append(d)
    return out
