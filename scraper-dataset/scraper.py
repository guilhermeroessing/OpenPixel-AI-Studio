#!/usr/bin/env python3
"""Coletor de assets de pixel art de fontes abertas (sem login) para o OpenPixel.

Uso: python scraper.py --limit 50

Por fonte: descobre candidatos -> baixa somente PNG < 2 MB (zips são lidos
em memória, nunca extraídos em disco nem executados) -> filtro OpenCV
(escala 1:1 via GCD, descarte de mixels/anti-aliasing/blur) -> classifica
(isolated_sprite / full_tile_scene) -> cataloga paleta -> salva PNG + legenda
.txt em processed/<fonte>/ e grava a linha no SQLite dataset_catalog.db.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import re
import sqlite3
import time
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable
from urllib import robotparser
from urllib.parse import quote, urljoin, urlparse

import cv2
import httpx
import numpy as np

UA = "OpenPixelDatasetBot/0.1"
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
DB_PATH = BASE_DIR / "dataset_catalog.db"

MAX_PNG_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 40 * 1024 * 1024
MAX_TEXT_BYTES = 3 * 1024 * 1024
MAX_DIM = 1024
MAX_CANDIDATES = 150  # teto de candidatos examinados por fonte
DEFAULT_DELAY = 1.5  # segundos entre requests ao mesmo host (cresce se o robots pedir Crawl-delay)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS assets("
    "id INTEGER PRIMARY KEY, source TEXT, path TEXT, caption_path TEXT, kind TEXT, "
    "width INT, height INT, scale INT, palette_json TEXT, approved INT, license TEXT, origin_url TEXT);"
    "CREATE INDEX IF NOT EXISTS idx_assets_source_origin ON assets(source, origin_url);"
)


class Blocked(Exception):
    """Host bloqueado por robots.txt ou por opt-out de IA (Content-Signal ai-train=no)."""


class FetchError(Exception):
    pass


class TooBig(Exception):
    pass


class Reject(Exception):
    """Asset reprovado pelo filtro; a mensagem é o motivo."""


@dataclass
class Item:
    origin_url: str
    license: str
    name: str
    data: bytes


@dataclass
class HostPolicy:
    parser: robotparser.RobotFileParser | None
    ai_optout: bool
    crawl_delay: float


class Http:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.client = httpx.AsyncClient(
            headers={"User-Agent": UA}, timeout=httpx.Timeout(30.0), follow_redirects=True
        )
        self._policies: dict[str, HostPolicy] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def _lock(self, host: str) -> asyncio.Lock:
        return self._locks.setdefault(host, asyncio.Lock())

    async def policy(self, url: str) -> HostPolicy:
        parts = urlparse(url)
        host = parts.netloc
        if host in self._policies:
            return self._policies[host]
        async with self._lock(host):
            if host in self._policies:
                return self._policies[host]
            try:
                r = await self.client.get(f"{parts.scheme}://{host}/robots.txt", timeout=20)
            except httpx.HTTPError as exc:
                raise Blocked(f"robots.txt de {host} indisponível ({type(exc).__name__})") from exc
            if r.status_code >= 500:
                raise Blocked(f"robots.txt de {host} retornou HTTP {r.status_code}")
            text = r.text if r.status_code == 200 else ""
            rp = None
            crawl_delay = 0.0
            if text:
                rp = robotparser.RobotFileParser()
                rp.parse(text.splitlines())
                try:
                    crawl_delay = float(rp.crawl_delay(UA) or 0)
                except (TypeError, ValueError):
                    crawl_delay = 0.0
            policy = HostPolicy(
                parser=rp,
                ai_optout=bool(re.search(r"Content-Signal:[^\n]*ai-train=no", text, re.I)),
                crawl_delay=crawl_delay,
            )
            self._policies[host] = policy
            return policy

    async def _do(self, url: str, headers: dict | None, max_bytes: int) -> tuple[int, bytes]:
        async with self.client.stream("GET", url, headers=headers) as r:
            if r.status_code >= 400:
                return r.status_code, b""
            cl = r.headers.get("content-length", "")
            if cl.isdigit() and int(cl) > max_bytes:
                raise TooBig(f"{int(cl)} bytes")
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise TooBig(f">{max_bytes} bytes")
            return r.status_code, bytes(buf)

    async def get(self, url: str, *, headers: dict | None = None, max_bytes: int = MAX_TEXT_BYTES) -> bytes:
        parts = urlparse(url)
        host = parts.netloc
        pol = await self.policy(url)
        if pol.ai_optout:
            raise Blocked(f"{host} declara Content-Signal ai-train=no (sem uso para treino de IA)")
        if pol.parser is not None and not pol.parser.can_fetch(UA, url):
            raise Blocked(f"robots.txt de {host} proíbe {parts.path}")
        gap = max(self.delay, pol.crawl_delay)
        async with self._lock(host):
            wait = self._last.get(host, 0.0) + gap - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            status, body = 0, b""
            for attempt in range(3):
                try:
                    status, body = await self._do(url, headers, max_bytes)
                except httpx.TransportError:
                    status, body = 0, b""
                self._last[host] = time.monotonic()
                if status in (429, 503) and attempt < 2:
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                break
        if status == 0:
            raise FetchError(f"falha de rede em {parts.path}")
        if status >= 400:
            raise FetchError(f"HTTP {status} em {parts.path}")
        return body

    async def text(self, url: str, headers: dict | None = None) -> str:
        return (await self.get(url, headers=headers)).decode("utf-8", "replace")

    async def json_(self, url: str, headers: dict | None = None):
        return json.loads(await self.text(url, headers))

    async def close(self) -> None:
        await self.client.aclose()


@dataclass
class Ctx:
    http: Http
    seen: set[str]
    log: Callable[[str], None]


# ---------------------------------------------------------------- licenças

def license_family(raw: str) -> str | None:
    up = raw.upper().replace("_", " ")
    if re.search(r"\b(NC|ND|SA|LGPL|LESSER|GFDL)\b|SHARE ?ALIKE|NON.?COMMERCIAL|NO.?DERIV", up):
        return None
    if "CC0" in up:
        return "CC0"
    if "PUBLIC DOMAIN" in up or "PUBLIC-DOMAIN" in up:
        return "PUBLIC-DOMAIN"
    if "OGA-BY" in up or "OGA BY" in up:
        return "OGA-BY"
    if re.search(r"\bCC[ -]?BY\b", up):
        return "CC-BY"
    if re.search(r"\bGPL\b|GPL[- ]?\d|GENERAL PUBLIC LICENSE", up):
        return "GPL"
    return None


def pick_license(options: list[str]) -> str | None:
    """Primeira licença aceita entre as opções (obras duais permitem escolher uma)."""
    for opt in options:
        if license_family(opt):
            return opt.strip()
    return None


# ---------------------------------------------------------------- filtro OpenCV

def detect_scale(rgba: np.ndarray) -> int:
    """Maior bloco inteiro compatível com todas as faixas uniformes (GCD dos runs)."""
    g = 0
    cols_changed = np.any(rgba[:, 1:] != rgba[:, :-1], axis=(0, 2))
    rows_changed = np.any(rgba[1:] != rgba[:-1], axis=(1, 2))
    for changed, length in ((cols_changed, rgba.shape[1]), (rows_changed, rgba.shape[0])):
        edges = np.concatenate(([0], np.flatnonzero(changed) + 1, [length]))
        for run in np.diff(edges):
            g = math.gcd(g, int(run))
            if g == 1:
                return 1
    return max(g, 1)


def _smooth_ratio(rgba: np.ndarray) -> float:
    """Fração de vizinhos opacos com diferença pequena mas não nula (sinal de blur/degradê)."""
    rgb = rgba[..., :3].astype(np.int16)
    op = rgba[..., 3] > 0
    dh = np.abs(rgb[:, 1:] - rgb[:, :-1]).max(axis=2)[op[:, 1:] & op[:, :-1]]
    dv = np.abs(rgb[1:] - rgb[:-1]).max(axis=2)[op[1:] & op[:-1]]
    diffs = np.concatenate([dh, dv])
    if diffs.size == 0:
        return 0.0
    return float(np.count_nonzero((diffs > 0) & (diffs <= 6)) / diffs.size)


@dataclass
class Processed:
    rgba: np.ndarray
    kind: str
    width: int
    height: int
    scale: int
    palette: list[str]
    n_colors: int


def process_png(data: bytes) -> Processed:
    if len(data) > MAX_PNG_BYTES:
        raise Reject("maior que 2 MB")
    if len(data) < 33 or not data.startswith(PNG_MAGIC):
        raise Reject("não é PNG")
    w0 = int.from_bytes(data[16:20], "big")
    h0 = int.from_bytes(data[20:24], "big")
    if not (4 <= w0 <= MAX_DIM and 4 <= h0 <= MAX_DIM):
        raise Reject(f"dimensões fora do limite ({w0}x{h0})")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Reject("PNG não decodificável")
    if img.dtype != np.uint8:
        raise Reject("profundidade de cor não suportada")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] != 4:
        raise Reject("canais não suportados")
    rgba = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    rgba[rgba[..., 3] == 0] = 0  # pixels transparentes não contam como cores

    scale = detect_scale(rgba)
    if scale > 1:
        rgba = np.ascontiguousarray(rgba[::scale, ::scale])
    h, w = rgba.shape[:2]
    if min(h, w) < 8:
        raise Reject(f"pequeno demais após redução 1:1 ({w}x{h})")
    if scale == 1 and np.all(rgba[..., 3] == 0):
        raise Reject("totalmente transparente")

    alpha = rgba[..., 3]
    keys = rgba.view(np.uint32)[..., 0]
    colors, counts = np.unique(keys, return_counts=True)
    n = int(colors.size)
    if n < 2:
        raise Reject("cor única")
    partial = np.count_nonzero((alpha > 0) & (alpha < 255)) / alpha.size
    if partial > 0.01:
        raise Reject("alpha parcial (anti-aliasing)")
    if n > 256:
        raise Reject(f"{n} cores (blur ou foto)")
    if n >= 24 and np.count_nonzero(counts <= 2) / n > 0.5:
        raise Reject("cores isoladas (mixels)")
    if n >= 8 and _smooth_ratio(rgba) > 0.08:
        raise Reject("transições suaves (blur)")

    transparent = np.count_nonzero(alpha == 0) / alpha.size
    kind = "isolated_sprite" if transparent >= 0.05 else "full_tile_scene"
    opaque_rgb = rgba[alpha > 0][:, :3]
    cols, cnt = np.unique(opaque_rgb, axis=0, return_counts=True)
    order = np.argsort(-cnt)[:32]
    palette = [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r, g, b in cols[order]]
    return Processed(rgba, kind, w, h, scale, palette, int(cols.shape[0]))


# ---------------------------------------------------------------- persistência

def save_asset(source: str, name: str, origin: str, lic: str, res: Processed) -> tuple[str, str]:
    folder = PROCESSED_DIR / source
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:60] or "asset"
    base = f"{slug}_{hashlib.sha1(origin.encode()).hexdigest()[:8]}"
    png_path = folder / f"{base}.png"
    ok, buf = cv2.imencode(".png", cv2.cvtColor(res.rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise Reject("falha ao reencodar PNG")
    png_path.write_bytes(buf.tobytes())
    kind_txt = "sprite isolado com fundo transparente" if res.kind == "isolated_sprite" else "tile/cena opaca"
    caption = (
        f"{name}\n"
        f"Fonte: {source}\n"
        f"Licença: {lic}\n"
        f"Origem (atribuição): {origin}\n"
        f"Tipo: {res.kind} ({kind_txt})\n"
        f"Dimensões 1:1: {res.width}x{res.height} px; escala original detectada: {res.scale}x\n"
        f"Paleta: {res.n_colors} cores; dominantes: {', '.join(res.palette[:8])}\n"
        "Legenda gerada por template automático (sem modelo multimodal); revisar antes de treinar.\n"
    )
    cap_path = folder / f"{base}.txt"
    cap_path.write_text(caption, encoding="utf-8")
    return png_path.relative_to(BASE_DIR).as_posix(), cap_path.relative_to(BASE_DIR).as_posix()


# ---------------------------------------------------------------- helpers de fonte

def unique(seq) -> list:
    return list(dict.fromkeys(seq))


def round_robin(paths: list[str]) -> list[str]:
    """Intercala pastas de primeiro nível para diversidade entre as amostras."""
    groups: dict[str, deque] = {}
    for p in sorted(paths):
        groups.setdefault(p.split("/")[0] if "/" in p else "", deque()).append(p)
    out: list[str] = []
    queues = [q for q in groups.values()]
    while queues:
        for q in queues:
            out.append(q.popleft())
        queues = [q for q in queues if q]
    return out


async def png_item(ctx: Ctx, url: str, origin: str, lic: str, name: str) -> Item | None:
    try:
        data = await ctx.http.get(url, max_bytes=MAX_PNG_BYTES)
    except (FetchError, TooBig) as exc:
        ctx.log(f"pulado {url}: {exc}")
        return None
    return Item(origin, lic, name, data)


async def zip_items(ctx: Ctx, url: str, origin: str, lic: str) -> AsyncIterator[Item]:
    """Lê PNGs de um zip em memória. Nada é extraído em disco nem executado."""
    try:
        blob = await ctx.http.get(url, max_bytes=MAX_ARCHIVE_BYTES)
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except (FetchError, TooBig, zipfile.BadZipFile) as exc:
        ctx.log(f"zip pulado {url}: {exc}")
        return
    with zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.lower().endswith(".png") or "__MACOSX" in name:
                continue
            if info.file_size > MAX_PNG_BYTES:
                continue
            member_origin = f"{origin}#{name}"
            if member_origin in ctx.seen:
                continue
            try:
                data = zf.read(info)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                ctx.log(f"membro ilegível {name}: {exc}")
                continue
            yield Item(member_origin, lic, Path(name).stem, data)


# ---------------------------------------------------------------- fontes

OGA = "https://opengameart.org"


async def src_opengameart(ctx: Ctx) -> AsyncIterator[Item]:
    query = "keys=pixel+art&field_art_type_tid%5B%5D=9&sort_by=count&sort_order=DESC"
    for page in range(10):
        listing = await ctx.http.text(f"{OGA}/art-search-advanced?{query}&page={page}")
        contents = unique(re.findall(r'href="(/content/[^"#?]+)"', listing))
        if not contents:
            break
        for path in contents:
            page_url = OGA + path
            try:
                detail = await ctx.http.text(page_url)
            except (FetchError, TooBig) as exc:
                ctx.log(f"página pulada {page_url}: {exc}")
                continue
            lic = pick_license(re.findall(r"license-name'>\s*([^<]+?)\s*<", detail))
            if lic is None:
                continue
            for href in unique(re.findall(r'href="([^"]+/sites/default/files/[^"]+)"', detail)):
                if "/styles/" in href:
                    continue
                url = urljoin(OGA, href)
                fname = url.rsplit("/", 1)[-1].split("?")[0]
                origin = f"{page_url}#{fname}"
                if origin in ctx.seen:
                    continue
                low = fname.lower()
                if low.endswith(".png"):
                    item = await png_item(ctx, url, origin, lic, Path(fname).stem)
                    if item:
                        yield item
                elif low.endswith(".zip"):
                    async for it in zip_items(ctx, url, page_url, lic):
                        yield it


async def src_kenney(ctx: Ctx) -> AsyncIterator[Item]:
    # kenney.nl/robots.txt: "Disallow: /assets/" e "/downloads/" para todos os bots.
    html = await ctx.http.text("https://kenney.nl/assets?q=pixel")
    slugs = unique(re.findall(r'href="(/assets/[^"/?#]+)"', html))
    ctx.log(f"{len(slugs)} páginas de asset listadas")
    for slug in slugs[:MAX_CANDIDATES]:
        await ctx.http.text("https://kenney.nl" + slug)  # levanta Blocked pelo robots
    return
    yield  # pragma: no cover


DCSS_RAW = "https://raw.githubusercontent.com/crawl/crawl/master/crawl-ref/source/rltiles/"
GH_JSON = {"Accept": "application/vnd.github+json"}


async def src_dcss(ctx: Ctx) -> AsyncIterator[Item]:
    api = "https://api.github.com/repos/crawl/crawl"
    root = await ctx.http.json_(f"{api}/contents/crawl-ref/source", GH_JSON)
    sha = next(e["sha"] for e in root if e["name"] == "rltiles")
    tree = await ctx.http.json_(f"{api}/git/trees/{sha}?recursive=1", GH_JSON)
    paths = [
        t["path"]
        for t in tree.get("tree", [])
        if t["type"] == "blob" and t["path"].lower().endswith(".png") and t.get("size", 0) <= MAX_PNG_BYTES
    ]
    lic = pick_license(["Public domain (RLTiles, conforme rltiles/license.txt do DCSS)"])
    for path in round_robin(paths):
        origin = f"https://github.com/crawl/crawl/blob/master/crawl-ref/source/rltiles/{path}"
        if origin in ctx.seen:
            continue
        item = await png_item(ctx, DCSS_RAW + quote(path), origin, lic, Path(path).stem)
        if item:
            yield item


LPC_REPO = "https://github.com/OpenGameArt/LiberatedPixelCup"


async def src_lpc(ctx: Ctx) -> AsyncIterator[Item]:
    api = "https://api.github.com/repos/OpenGameArt/LiberatedPixelCup"
    tree = await ctx.http.json_(f"{api}/git/trees/master?recursive=1", GH_JSON)
    paths = [
        t["path"]
        for t in tree.get("tree", [])
        if t["type"] == "blob"
        and t["path"].lower().endswith(".png")
        and t["path"].startswith(("sprite/", "tileset/"))
        and t.get("size", 0) <= MAX_PNG_BYTES
    ]
    # README: dupla licença CC-BY-SA 3.0 / GPL 3.0; escolhemos GPL-3.0 (CC-BY-SA é reprovada).
    lic = pick_license(["GPL-3.0"])
    raw = "https://raw.githubusercontent.com/OpenGameArt/LiberatedPixelCup/master/"
    for path in round_robin(paths):
        origin = f"{LPC_REPO}/blob/master/{path}"
        if origin in ctx.seen:
            continue
        item = await png_item(ctx, raw + quote(path), origin, lic, Path(path).stem)
        if item:
            yield item


async def src_freegameassets(ctx: Ctx) -> AsyncIterator[Item]:
    # freegameassets.com/robots.txt: Content-Signal ai-train=no -> Blocked na primeira request.
    await ctx.http.text("https://freegameassets.com/")
    return
    yield  # pragma: no cover


async def src_lospec(ctx: Ctx) -> AsyncIterator[Item]:
    base = "https://lospec.com"
    artworks: list[str] = []
    for page in range(1, 4):
        url = f"{base}/gallery/" if page == 1 else f"{base}/gallery/?page={page}"
        try:
            html = await ctx.http.text(url)
        except (FetchError, TooBig):
            break
        artworks += re.findall(r'href="(/gallery/[^/"?#]+/[^/"?#]+)"', html)
    artworks = unique(artworks)[:MAX_CANDIDATES]
    ctx.log(f"{len(artworks)} obras listadas")
    no_license = 0
    for path in artworks:
        page_url = base + path
        try:
            html = await ctx.http.text(page_url)
        except (FetchError, TooBig):
            continue
        tokens = re.findall(
            r"CC0|CC[ -]BY(?:[ -]SA)?[ -]?\d(?:\.\d)?|OGA-BY[ -]?\d\.\d|GPL[ -]?\d(?:\.\d)?|public domain",
            html,
            re.I,
        )
        lic = pick_license(tokens)
        if lic is None:
            no_license += 1
            continue
        m = re.search(r'property="og:image"[^>]*content="([^"]+)"|content="([^"]+)"[^>]*property="og:image"', html)
        if not m:
            continue
        img = m.group(1) or m.group(2)
        if page_url in ctx.seen:
            continue
        item = await png_item(ctx, urljoin(base, img), page_url, lic, Path(path).name)
        if item:
            yield item
    ctx.log(f"{no_license} obras sem licença livre declarada (puladas)")


COMMONS = "https://commons.wikimedia.org/w/api.php"


async def src_wikimedia(ctx: Ctx) -> AsyncIterator[Item]:
    for term in ("pixel art", "pixel art sprite", "pixel art tile"):
        offset = 0
        for _ in range(8):
            q = (
                f"{COMMONS}?action=query&list=search&srnamespace=6&srlimit=50"
                f"&srsearch={quote(term)}&sroffset={offset}&format=json"
            )
            data = await ctx.http.json_(q)
            hits = data.get("query", {}).get("search", [])
            titles = [h["title"] for h in hits]
            for i in range(0, len(titles), 10):
                chunk = titles[i : i + 10]
                info_url = (
                    f"{COMMONS}?action=query&prop=imageinfo&iiprop=url|size|mime|extmetadata"
                    f"&iiextmetadatafilter=LicenseShortName&format=json&titles={quote('|'.join(chunk))}"
                )
                info = await ctx.http.json_(info_url)
                for page in info.get("query", {}).get("pages", {}).values():
                    ii = (page.get("imageinfo") or [None])[0]
                    if not ii or ii.get("mime") != "image/png" or ii.get("size", 0) > MAX_PNG_BYTES:
                        continue
                    raw_lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
                    lic = pick_license([raw_lic])
                    if lic is None:
                        continue
                    origin = ii["descriptionurl"]
                    if origin in ctx.seen:
                        continue
                    name = page["title"].split(":", 1)[-1].rsplit(".", 1)[0]
                    item = await png_item(ctx, ii["url"], origin, lic, name)
                    if item:
                        yield item
            if "continue" not in data or len(hits) == 0:
                break
            offset = data["continue"].get("sroffset", offset + 50)


SOURCES: dict[str, Callable[[Ctx], AsyncIterator[Item]]] = {
    "opengameart": src_opengameart,
    "kenney": src_kenney,
    "dcss-rltiles": src_dcss,
    "liberated-pixel-cup": src_lpc,
    "freegameassets": src_freegameassets,
    "lospec": src_lospec,
    "wikimedia-commons": src_wikimedia,
}


# ---------------------------------------------------------------- pipeline

@dataclass
class Report:
    source: str
    approved: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


async def run_source(key: str, adapter, http: Http, db: sqlite3.Connection, limit: int, log) -> Report:
    rep = Report(key)
    approved = db.execute("SELECT COUNT(*) FROM assets WHERE source=? AND approved=1", (key,)).fetchone()[0]
    seen = {r[0] for r in db.execute("SELECT origin_url FROM assets WHERE source=?", (key,)) if r[0]}
    rep.approved = approved
    if approved >= limit:
        rep.notes.append("já completo no banco")
        return rep
    ctx = Ctx(http, seen, lambda m: log(f"[{key}] {m}"))
    examined = 0
    gen = adapter(ctx)
    try:
        async for item in gen:
            if rep.approved >= limit:
                break
            if item.origin_url in seen:
                continue
            seen.add(item.origin_url)
            examined += 1
            if examined > MAX_CANDIDATES:
                rep.notes.append(f"teto de {MAX_CANDIDATES} candidatos atingido")
                break
            try:
                res = process_png(item.data)
            except Reject as exc:
                rep.rejected += 1
                db.execute(
                    "INSERT INTO assets(source, path, caption_path, kind, width, height, scale, palette_json, "
                    "approved, license, origin_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (key, None, None, None, None, None, None, None, 0, item.license, item.origin_url),
                )
                db.commit()
                log(f"reprovado {item.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - um arquivo ruim não derruba a fonte
                rep.rejected += 1
                log(f"erro no filtro {item.name}: {type(exc).__name__}: {exc}")
                continue
            path, cap = save_asset(key, item.name, item.origin_url, item.license, res)
            db.execute(
                "INSERT INTO assets(source, path, caption_path, kind, width, height, scale, palette_json, "
                "approved, license, origin_url) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key, path, cap, res.kind, res.width, res.height, res.scale,
                    json.dumps({"colors": res.palette, "total_colors": res.n_colors}),
                    1, item.license, item.origin_url,
                ),
            )
            db.commit()
            rep.approved += 1
            log(f"aprovado {rep.approved}/{limit}: {item.name} ({res.kind}, {res.width}x{res.height}, escala {res.scale}x)")
    except Blocked as exc:
        rep.errors.append(f"bloqueado: {exc}")
        log(f"bloqueado: {exc}")
    except Exception as exc:  # noqa: BLE001 - registra e segue para as outras fontes
        rep.errors.append(f"{type(exc).__name__}: {exc}")
        log(f"falhou: {type(exc).__name__}: {exc}")
    finally:
        if hasattr(gen, "aclose"):
            await gen.aclose()
    return rep


async def main_async(args) -> int:
    chosen = [s.strip() for s in args.sources.split(",")] if args.sources else list(SOURCES)
    unknown = [s for s in chosen if s not in SOURCES]
    if unknown:
        print(f"fontes desconhecidas: {unknown}")
        return 2
    PROCESSED_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    http = Http(args.delay)
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    try:
        reports = await asyncio.gather(
            *(run_source(k, SOURCES[k], http, db, args.limit, log) for k in chosen)
        )
    finally:
        await http.close()
        db.close()
    print("\nResumo por fonte (aprovados / reprovados):")
    for r in reports:
        extra = "; ".join(r.errors + r.notes)
        print(f"  {r.source:22s} {r.approved:3d} / {r.rejected:4d}  {extra}")
    failed = [r.source for r in reports if r.errors]
    print(f"Fontes com falha/bloqueio: {', '.join(failed) if failed else 'nenhuma'}")
    print(f"Total aprovado: {sum(r.approved for r in reports)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Coletor de pixel art de fontes abertas")
    parser.add_argument("--limit", type=int, default=50, help="máximo de assets APROVADOS por fonte")
    parser.add_argument("--sources", default="", help="lista separada por vírgula (padrão: todas)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="segundos entre requests por host")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
