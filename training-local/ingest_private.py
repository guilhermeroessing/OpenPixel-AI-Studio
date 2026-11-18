#!/usr/bin/env python3
"""Ingestor PRIVADO: referencias do usuario -> dataset local de treino.

A SAIDA (training-local/private/) NUNCA deve ser versionada (ver .gitignore).
Este script apenas le bytes de PNG/GIF com Pillow/numpy. Nao executa, nao
extrai arquivos e nao interpreta nada que venha das referencias.

Uso (a partir da raiz do projeto):
    python -E -P training-local/ingest_private.py
    python -E -P training-local/ingest_private.py --cap 300

Pipeline: varredura -> filtro de pixel art -> dedupe (sha1) -> limite por
pacote -> classificacao -> analise de pixels -> escrita (PNG, legenda, SQLite).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "references-local"
PRIVATE = ROOT / "training-local" / "private"
IMG_DIR = PRIVATE / "images"
DB_PATH = PRIVATE / "private_catalog.db"
REPORT_PATH = PRIVATE / "ingest_report.json"
SCRAPER_DIR = ROOT / "scraper-dataset"

# Reuso das funcoes de pixel do scraper (sem executar o main dele).
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRAPER_DIR))
import scraper as sc  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LICENSE_NOTE = "fornecido pelo usuario; uso local para treino; sem redistribuicao"
PACK_CAP = 300
IMG_EXTS = {".png", ".gif"}
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_SIDE = 4096
MAX_FRAMES = 256
ANALYSIS_MAX_SIDE = 2048  # imagens maiores sao analisadas em versao reduzida (NEAREST)
ALPHA_PARTIAL_MAX = 0.01  # acima disso ha anti-aliasing/glow
COLOR_MAX = 256
COLOR_MAX_ANIM = 512  # GIF: paleta por frame, uniao pode passar de 256
SMOOTH_MAX = 0.08  # razao de transicoes suaves (blur/degrade)
MIN_SIDE = 8
SPRITE_MIN_TRANSPARENT = 0.05

# Pacotes excluidos (licenca proibe IA/comercial ou desconhecida). Comparados sem espacos/pontuacao, minusculo.
EXCLUDED_PACK_KEYS = ("freelandscapes", "postapocalypse", "scifiplatformer")
EXCLUDED_STEM_SUBSTR = ("screenshot", "thumbnail", "preview", "cover")
EXCLUDED_STEM_EXACT = ("icon", "thumb", "thumbs")
EXCLUDED_REL_PREFIX = (("referencias", "mapas"),)

# Heuristica de categoria (tokens exatos ou substrings longas).
PAR_TOK = {"bg", "sky", "layer", "layers", "back", "far", "hill", "hills"}
PAR_SUB = ("parallax", "background", "backdrop")
SCEN_TOK = {"forest", "city", "cave", "space", "mountain", "mountains", "ocean", "desert", "winter", "dusk",
            "planet", "planets", "cliff", "cliffs", "wasteland", "wastelands", "landscape", "landscapes",
            "underwater", "village", "sunnyland", "scenery"}
SCEN_SUB = ("background", "parallax", "landscape", "mountain", "wasteland", "underwater", "scene")
EFF_TOK = {"fx", "hit", "bolt", "beam", "blast", "spark", "sparks", "aura", "glow", "smoke", "splash", "burst",
           "projectile", "impact", "slash", "effect", "effects", "holy", "fire", "flame", "flames", "lightning",
           "magic", "spell", "spells", "blood", "skill", "skills", "particle", "particles", "explosion",
           "explosions", "poof"}
EFF_SUB = ("explo", "particle", "projectile", "lightning", "splash", "impact", "spell")
EFF_WEAK_TOK = {"water", "void", "dark"}  # so vale para sprite isolado e depois de checar personagem
ENV_TOK = {"tileset", "tilesets", "tile", "tiles", "tilemap", "walls", "wall", "floor", "ground", "terrain",
           "grass", "dirt", "rock", "rocks", "brick", "bricks", "stone", "stones", "bridge", "platform",
           "platforms", "decor", "door", "doors", "cave", "environment", "dungeon", "outdoor", "indoor", "ts",
           "statics", "tilemap"}
ENV_SUB = ("tileset", "environment", "terrain", "dungeon")
ASSET_TOK = {"item", "items", "weapon", "weapons", "armour", "armor", "armours", "potion", "potions", "shield",
             "shields", "sword", "swords", "chest", "chests", "coin", "coins", "gem", "gems", "key", "keys",
             "prop", "props", "icon", "icons", "tool", "tools", "food", "hat", "hats", "boots", "bow", "bows",
             "staff", "axe", "ui"}
ASSET_SUB = ("weapon", "potion", "armour", "armor")
CHAR_TOK = {"char", "chars", "character", "characters", "personagem", "personagens", "monster", "monsters",
            "monstro", "monstros", "mon", "mons", "enemy", "enemies", "npc", "player", "hero", "knight",
            "creature", "creatures", "humanoid", "idle", "walk", "run", "attack", "hurt", "death", "jump",
            "cast", "pony", "skeleton", "zombie", "goblin", "orc"}
CHAR_SUB = ("character", "monster", "creature")
PACK_EFF_TOK = {"fx", "fire", "explosion", "explosions", "lightning", "skill", "skills", "magic", "blood", "spell"}

CAT_EN = {
    "character": "Character sprite",
    "environment": "Environment tile/scene",
    "asset": "Item/asset sprite",
    "effect": "Effect sprite",
    "terrain_parallax": "Parallax background layer",
}


class Reject(Exception):
    pass


@dataclass
class Cand:
    path: Path
    rel: str
    pack: str
    item: str
    category: str
    kind: str
    width: int
    height: int
    frames: int
    is_anim: bool
    durations: list
    loop: object
    n_colors: int
    sha1: str
    priority: int


# ---------------------------------------------------------------- varredura e exclusoes

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def tokens_of(text: str) -> tuple[str, set]:
    low = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
    return low, {t for t in re.split(r"[^a-z0-9]+", low) if t}


def has(tok_set: set, sub_words: tuple, toks: set, low: str) -> bool:
    return bool(toks & tok_set) or any(s in low for s in sub_words)


def pack_of(parts: tuple) -> str | None:
    """Nome do pacote a partir do caminho relativo a references-local."""
    if len(parts) >= 2 and parts[0] == "referencias" and parts[1] == "download itchio":
        base, prefix, root_name = 2, "", "download itchio (raiz)"
    elif len(parts) >= 1 and parts[0] == "extra":
        base, prefix, root_name = 1, "extra:", "extra (raiz)"
    else:
        return None  # arquivo solto em referencias/ (ex.: paleta.png): fora de pacote
    if len(parts) > base + 1:
        return prefix + parts[base]
    return root_name


def excluded_dir_reason(parts: tuple) -> str | None:
    if "__MACOSX" in parts:
        return "__MACOSX"
    if parts[:2] == ("referencias", "mapas"):
        return "referencias/mapas (jogos comerciais)"
    n = norm(parts[-1])
    for key in EXCLUDED_PACK_KEYS:
        if key in n:
            return f"pacote excluido (licenca proibe IA/comercial ou desconhecida): {parts[-1]}"
    return None


def excluded_file_reason(parts: tuple) -> str | None:
    if any(p == "__MACOSX" for p in parts):
        return "__MACOSX"
    stem = Path(parts[-1]).stem
    s = norm(stem)
    if stem.lower() in EXCLUDED_STEM_EXACT or any(k in s for k in EXCLUDED_STEM_SUBSTR):
        return "icon/screenshot/preview/cover/thumbnail"
    return None


def count_images(folder: Path) -> int:
    total = 0
    for _, _, fns in os.walk(folder):
        total += sum(1 for f in fns if os.path.splitext(f)[1].lower() in IMG_EXTS)
    return total


@dataclass
class Stats:
    excluded_dirs: list
    excluded_files: Counter
    non_image: Counter
    rejected_by_pack: Counter
    reject_reasons: Counter
    erros: int = 0


def walk_refs(st: Stats) -> list[tuple[Path, tuple]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(REFS):
        dparts = Path(dirpath).relative_to(REFS).parts
        keep = []
        for d in dirnames:
            full = Path(dirpath) / d
            parts = dparts + (d,)
            if full.is_symlink():
                continue
            reason = excluded_dir_reason(parts)
            if reason:
                st.excluded_dirs.append({"pasta": "/".join(parts), "motivo": reason,
                                         "png_gif": count_images(full)})
                continue
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            full = Path(dirpath) / f
            parts = dparts + (f,)
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMG_EXTS:
                st.non_image[ext or "<sem extensao>"] += 1
                continue
            if full.is_symlink():
                continue
            reason = excluded_file_reason(parts)
            if reason is None and pack_of(parts) is None:
                reason = "fora de pacote (arquivo solto em referencias/)"
            if reason:
                st.excluded_files[reason] += 1
                continue
            out.append((full, parts))
    out.sort(key=lambda t: t[1])
    return out


# ---------------------------------------------------------------- filtro de pixel art

def read_normalized(path: Path) -> tuple[np.ndarray, dict]:
    """Decodifica PNG/GIF/APNG como RGBA 1:1. Animacoes viram spritesheet horizontal."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise Reject("arquivo maior que 30 MB")
    with Image.open(path) as im:
        if im.format not in ("PNG", "GIF"):
            raise Reject(f"formato nao suportado ({im.format})")
        n = int(getattr(im, "n_frames", 1) or 1)
        W, H = im.size
        if not (4 <= W <= MAX_SIDE and 4 <= H <= MAX_SIDE):
            raise Reject(f"dimensoes fora do limite ({W}x{H})")
        if n > 1:
            if n > MAX_FRAMES:
                raise Reject(f"animacao com mais de {MAX_FRAMES} frames")
            frames, durs = [], []
            for fr in ImageSequence.Iterator(im):
                durs.append(int(fr.info.get("duration") or im.info.get("duration") or 100))
                frames.append(np.array(fr.convert("RGBA")))
            if len({f.shape for f in frames}) != 1:
                raise Reject("frames de tamanhos diferentes")
            rgba = np.concatenate(frames, axis=1)
            info = {"is_anim": True, "frames": n, "durations": durs, "loop": im.info.get("loop")}
        else:
            rgba = np.array(im.convert("RGBA"))
            info = {"is_anim": False, "frames": 1, "durations": [], "loop": None}
    rgba = np.ascontiguousarray(rgba)
    rgba[rgba[..., 3] == 0] = 0  # transparente nao conta como cor
    scale = sc.detect_scale(rgba)
    if scale > 1:
        rgba = np.ascontiguousarray(rgba[::scale, ::scale])
    return rgba, info


def pixel_art_filter(rgba: np.ndarray, is_anim: bool) -> str:
    h, w = rgba.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise Reject("pequeno demais apos reducao 1:1")
    alpha = rgba[..., 3]
    if np.all(alpha == 0):
        raise Reject("totalmente transparente")
    partial = np.count_nonzero((alpha > 0) & (alpha < 255)) / alpha.size
    if partial > ALPHA_PARTIAL_MAX:
        raise Reject("alpha parcial (anti-aliasing/glow)")
    keys = rgba.view(np.uint32)[..., 0]
    colors, counts = np.unique(keys, return_counts=True)
    n = int(colors.size)
    if n < 2:
        raise Reject("cor unica")
    limit = COLOR_MAX_ANIM if is_anim else COLOR_MAX
    if n > limit:
        raise Reject(f"muitas cores (foto/gradiente)")
    if n >= 24 and np.count_nonzero(counts <= 2) / n > 0.5:
        raise Reject("cores isoladas (mixels)")
    if n >= 8 and sc._smooth_ratio(rgba) > SMOOTH_MAX:
        raise Reject("transicoes suaves (blur/degrade)")
    transparent = np.count_nonzero(alpha == 0) / alpha.size
    return "isolated_sprite" if transparent >= SPRITE_MIN_TRANSPARENT else "full_tile_scene"


# ---------------------------------------------------------------- classificacao

def classify(item: str, pack: str, w: int, h: int, kind: str) -> str:
    ftext, ftok = tokens_of(item)
    ptext, ptok = tokens_of(pack)
    wide = w >= 2 * h
    if has(PAR_TOK, PAR_SUB, ftok, ftext):
        return "terrain_parallax"
    if kind == "full_tile_scene":
        if has(set(), PAR_SUB, ptok, ptext):
            return "terrain_parallax"
        if wide and has(SCEN_TOK, SCEN_SUB, ptok, ptext):
            return "terrain_parallax"
    if has(EFF_TOK, EFF_SUB, ftok, ftext):
        return "effect"
    if has(ENV_TOK, ENV_SUB, ftok, ftext):
        return "environment"
    if has(ASSET_TOK, ASSET_SUB, ftok, ftext):
        return "asset"
    if has(CHAR_TOK, CHAR_SUB, ftok, ftext):
        return "character"
    if kind == "isolated_sprite" and has(EFF_WEAK_TOK, (), ftok, ftext):
        return "effect"
    # Pistas no nome do pacote (so quando o arquivo nao diz nada)
    if has(PACK_EFF_TOK, (), ptok, ptext):
        return "effect"
    if has(ASSET_TOK, ASSET_SUB, ptok, ptext):
        return "asset"
    if kind == "isolated_sprite" and has(CHAR_TOK, CHAR_SUB, ptok, ptext):
        return "character"
    if kind == "full_tile_scene":
        return "environment"
    return "asset"


def priority(c_is_anim: bool, kind: str, w: int, h: int, item: str) -> int:
    stem = item.rsplit("/", 1)[-1]
    if c_is_anim:
        return 0
    if kind == "isolated_sprite" and w >= 2 * h and w >= 64:
        return 0  # tira horizontal (provavel spritesheet)
    if kind == "full_tile_scene" and w * h >= 128 * 128:
        return 1  # fundo/cena completa
    if re.search(r"(?:^|[_\- ])\d{1,4}$", stem):
        return 3  # frame solto numerado: redundante
    return 2


def build_candidate(path: Path, parts: tuple, pack: str) -> Cand:
    rgba, info = read_normalized(path)
    kind = pixel_art_filter(rgba, info["is_anim"])
    h, w = rgba.shape[:2]
    base = 2 if parts[0] == "referencias" else 1
    inside = parts[base + 1:] if len(parts) > base + 1 else parts[base:]
    item = "/".join(inside[:-1] + (Path(inside[-1]).stem,))
    category = classify(item, pack, w, h, kind)
    sha1 = hashlib.sha1(rgba.tobytes() + f"{w}x{h}".encode()).hexdigest()
    return Cand(
        path=path, rel="/".join(parts), pack=pack, item=item, category=category, kind=kind,
        width=w, height=h, frames=info["frames"], is_anim=info["is_anim"], durations=info["durations"],
        loop=info["loop"], n_colors=int(np.unique(rgba.view(np.uint32)[..., 0]).size), sha1=sha1,
        priority=priority(info["is_anim"], kind, w, h, item),
    )


# ---------------------------------------------------------------- analise e legenda

def analyze(rgba_full: np.ndarray, c: Cand) -> dict:
    H, W = rgba_full.shape[:2]
    s = max(W, H)
    ref = rgba_full
    scale = 1.0
    if s > ANALYSIS_MAX_SIDE:
        scale = ANALYSIS_MAX_SIDE / s
        nw, nh = max(1, round(W * scale)), max(1, round(H * scale))
        ref = np.ascontiguousarray(np.array(Image.fromarray(rgba_full).resize((nw, nh), Image.NEAREST)))
    a = sc.analyze_pixels(ref)
    a["analysis_scale"] = round(scale, 4)
    a["category"] = c.category
    a["kind"] = c.kind
    a["pack"] = c.pack
    a["animation"] = None
    if c.is_anim:
        n = c.frames
        fw = W // n
        frame0 = np.ascontiguousarray(rgba_full[:, :fw])
        a0 = sc.analyze_pixels(frame0)
        a["tile_grid"] = a0["tile_grid"]
        a["dark_outline"] = a0["dark_outline"]
        a["frames"], a["frames_cols"], a["frames_rows"] = n, n, 1
        a["spritesheet"] = True
        a["frames_uniform"] = True
        total = int(sum(c.durations))
        seamless = bool(np.array_equal(frame0, rgba_full[:, (n - 1) * fw:]))
        a["animation"] = {
            "frames": n,
            "durations_ms": c.durations,
            "total_ms": total,
            "fps": round(1000 * n / total, 2) if total else None,
            "loop_flag": c.loop,
            "first_equals_last_frame": seamless,
        }
    return a


def caption_for(c: Cand, a: dict) -> str:
    kind_en = "sprite with transparent background" if c.kind == "isolated_sprite" else "opaque tile/scene"
    item_readable = c.item.replace("/", " / ").replace("_", " ").replace("-", " ")
    lines = [
        f"{CAT_EN[c.category]} pixel art: {item_readable}.",
        f"Pack: {c.pack}. Size {c.width}x{c.height} px, {kind_en}, {c.n_colors} colors, "
        "hard pixel edges (no anti-aliasing).",
    ]
    if a["animation"]:
        an = a["animation"]
        loop = {0: "loops forever", None: "loop flag not set"}.get(an["loop_flag"], f"loop count {an['loop_flag']}")
        seam = "seamless loop" if an["first_equals_last_frame"] else "not a seamless loop"
        lines.append(f"Animation: {an['frames']} frames in a horizontal spritesheet, "
                     f"{an['fps']} fps, {loop}, {seam}.")
    lines.append("Dominant colors: " + (", ".join(a["dominant_colors"]) or "none") + ".")
    text = "\n".join(lines) + "\n\n" + sc.render_analysis(a)
    extra = []
    if a["animation"]:
        an = a["animation"]
        extra.append(f"- Animacao: {an['frames']} frames; duracao total {an['total_ms']} ms; "
                     f"{an['fps']} fps; loop do arquivo: {an['loop_flag']}; "
                     f"primeiro frame igual ao ultimo: {'sim' if an['first_equals_last_frame'] else 'nao'}")
    if a["analysis_scale"] != 1.0:
        extra.append(f"- Analise feita em versao reduzida ({a['analysis_scale']}x, NEAREST) por imagem > "
                     f"{ANALYSIS_MAX_SIDE} px")
    if extra:
        text += "\n" + "\n".join(extra)
    text += "\nLegenda visual pendente: sim (descricao visual a ser feita por agente)\n"
    return text


def slug(text: str, maxlen: int = 90) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s or "x")[:maxlen].rstrip("-") or "x"


# ---------------------------------------------------------------- principal

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingestor privado de referencias para treino local.")
    parser.add_argument("--cap", type=int, default=PACK_CAP, help="limite de imagens por pacote")
    args = parser.parse_args()

    if not REFS.is_dir():
        print(f"[ERRO] pasta de referencias nao encontrada: {REFS}")
        return 1
    if PRIVATE.name != "private":
        print("[ERRO] destino inesperado")
        return 1

    t0 = time.time()
    st = Stats([], Counter(), Counter(), Counter(), Counter())

    # 1) varredura
    files = walk_refs(st)
    print(f"[1/5] varredura: {len(files)} PNG/GIF elegiveis; "
          f"{sum(st.excluded_files.values())} arquivos e {len(st.excluded_dirs)} pastas excluidos", flush=True)

    # 2) filtro de pixel art (leitura apenas)
    cands: list[Cand] = []
    for i, (path, parts) in enumerate(files, 1):
        pack = pack_of(parts)
        try:
            cands.append(build_candidate(path, parts, pack))
        except Reject as e:
            st.rejected_by_pack[pack] += 1
            st.reject_reasons[str(e)] += 1
        except Exception as e:  # imagem corrompida / bomba de descompressao
            st.rejected_by_pack[pack] += 1
            st.reject_reasons[f"erro de decodificacao ({type(e).__name__})"] += 1
        if i % 1000 == 0:
            print(f"    filtro: {i}/{len(files)} lidos, {len(cands)} aprovados ({time.time() - t0:.0f}s)", flush=True)
    print(f"[2/5] filtro pixel art: {len(cands)} aprovados de {len(files)}", flush=True)

    # 3) dedupe global por sha1 do conteudo normalizado
    seen: dict[str, Cand] = {}
    dup_by_pack: Counter = Counter()
    dup_cross: Counter = Counter()
    unique: list[Cand] = []
    for c in sorted(cands, key=lambda c: c.rel):
        first = seen.get(c.sha1)
        if first is not None:
            dup_by_pack[c.pack] += 1
            if first.pack != c.pack:
                dup_cross[c.pack] += 1
            continue
        seen[c.sha1] = c
        unique.append(c)
    print(f"[3/5] dedupe: {len(unique)} unicos; {len(cands) - len(unique)} duplicatas removidas", flush=True)

    # 4) limite por pacote, preferindo animacoes/spritesheets/fundos
    by_pack: dict[str, list[Cand]] = defaultdict(list)
    for c in unique:
        by_pack[c.pack].append(c)
    selected: list[Cand] = []
    over_cap: Counter = Counter()
    for pack, group in by_pack.items():
        group.sort(key=lambda c: (c.priority, c.item))
        selected.extend(group[: args.cap])
        if len(group) > args.cap:
            over_cap[pack] = len(group) - args.cap
    print(f"[4/5] limite {args.cap}/pacote: {len(selected)} selecionados; {sum(over_cap.values())} descartados por limite",
          flush=True)

    # 5) analise, legenda e escrita
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    if DB_PATH.exists():
        DB_PATH.unlink()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute(
        "CREATE TABLE assets (id INTEGER PRIMARY KEY, pack TEXT NOT NULL, rel_path TEXT NOT NULL, "
        "category TEXT NOT NULL, width INTEGER, height INTEGER, frames INTEGER, tile_size INTEGER, "
        "colors INTEGER, has_outline INTEGER, analysis_json TEXT, caption_path TEXT, hash TEXT NOT NULL, "
        "license_note TEXT NOT NULL, image_path TEXT NOT NULL)"
    )
    db.execute("CREATE UNIQUE INDEX ux_assets_hash ON assets(hash)")
    db.execute("CREATE INDEX ix_assets_cat ON assets(category)")

    per_cat: Counter = Counter()
    per_pack_cat: Counter = Counter()
    used_names: set[str] = set()
    for i, c in enumerate(selected, 1):
        rgba, _ = read_normalized(c.path)
        a = analyze(rgba, c)
        pack_slug = slug(c.pack.replace(":", "-"))
        item_slug = slug(c.item.replace("/", "-"))
        name = f"{pack_slug}__{item_slug}"
        n_try, final = 2, name
        while (c.category, final) in used_names or (IMG_DIR / c.category / f"{final}.png").exists():
            final = f"{name}-{n_try}"
            n_try += 1
        used_names.add((c.category, final))
        cat_dir = IMG_DIR / c.category
        cat_dir.mkdir(parents=True, exist_ok=True)
        img_path = cat_dir / f"{final}.png"
        cap_path = cat_dir / f"{final}.txt"
        Image.fromarray(rgba).save(img_path, optimize=True)
        cap_path.write_text(caption_for(c, a), encoding="utf-8")
        tile = a["tile_grid"]
        db.execute(
            "INSERT INTO assets (pack, rel_path, category, width, height, frames, tile_size, colors, has_outline, "
            "analysis_json, caption_path, hash, license_note, image_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.pack, c.rel, c.category, c.width, c.height, c.frames, tile, a["n_colors"],
             1 if a["dark_outline"] else 0, json.dumps(a, ensure_ascii=False, default=str),
             cap_path.relative_to(PRIVATE).as_posix(), c.sha1, LICENSE_NOTE,
             img_path.relative_to(PRIVATE).as_posix()),
        )
        per_cat[c.category] += 1
        per_pack_cat[(c.pack, c.category)] += 1
        if i % 200 == 0:
            print(f"    escrita: {i}/{len(selected)} ({time.time() - t0:.0f}s)", flush=True)
    db.commit()
    db.close()
    print(f"[5/5] escrita concluida: {len(selected)} imagens em {IMG_DIR}", flush=True)

    per_pack_total = Counter({p: len(g) for p, g in by_pack.items()})
    pack_sel = Counter(c.pack for c in selected)
    report = {
        "total_ingerido": len(selected),
        "por_categoria": dict(per_cat),
        "categorias_acima_de_100": {k: v for k, v in per_cat.items() if v > 100},
        "pacotes_pair_categoria_acima_de_100": {f"{p} / {c}": v for (p, c), v in per_pack_cat.items() if v > 100},
        "por_pacote": {p: {"ingeridos": pack_sel[p], "unicos_antes_do_limite": per_pack_total[p],
                           "duplicatas_removidas": dup_by_pack[p],
                           "duplicatas_de_outro_pacote": dup_cross[p],
                           "rejeitados_filtro": st.rejected_by_pack[p],
                           "descartados_por_limite": over_cap[p]}
                       for p in sorted(set(pack_sel) | set(per_pack_total) | set(st.rejected_by_pack))},
        "excluidos_arquivos": dict(st.excluded_files),
        "excluidos_pastas": st.excluded_dirs,
        "ignorados_nao_imagem": dict(st.non_image.most_common(40)),
        "motivos_rejeicao": dict(st.reject_reasons.most_common()),
        "segundos": round(time.time() - t0, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"TOTAL: {len(selected)} | por categoria: {dict(per_cat)}")
    print(f"relatorio: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
