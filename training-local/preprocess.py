"""Pre-processamento para treino LoRA: upscale NEAREST para 1024px.

Le imagens de scraper-dataset/processed que tenham legenda .txt de mesmo nome,
centraliza cada uma em canvas quadrado (sem distorcer), amplia com
Image.NEAREST para RESOLUTION x RESOLUTION e copia a legenda junto.

Uso (a partir da raiz do projeto):
    python training-local/preprocess.py
    python training-local/preprocess.py --src scraper-dataset/processed --out training-local/data/processed_1024
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def hex_to_rgba(value: str) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("--bg deve ser hex RRGGBB")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def square_pad(img: Image.Image, bg: tuple[int, int, int, int]) -> Image.Image:
    side = max(img.width, img.height)
    canvas = Image.new("RGBA", (side, side), bg)
    offset = ((side - img.width) // 2, (side - img.height) // 2)
    canvas.alpha_composite(img, offset)
    return canvas


def process(img: Image.Image, resolution: int, bg: tuple[int, int, int, int]) -> Image.Image:
    img = square_pad(img.convert("RGBA"), bg)
    img = img.resize((resolution, resolution), Image.NEAREST)
    return img.convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale NEAREST para treino SDXL LoRA.")
    parser.add_argument("--src", type=Path, default=ROOT / "scraper-dataset" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT / "training-local" / "data" / "processed_1024")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--bg", default="000000", help="cor de fundo do padding, hex RRGGBB")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"[ERRO] Pasta de origem nao encontrada: {args.src}")
        return 1

    bg = hex_to_rgba(args.bg)
    args.out.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    done = 0
    for img_path in images:
        caption = img_path.with_suffix(".txt")
        if not caption.exists():
            print(f"[SKIP] sem legenda: {img_path}")
            continue

        # Achata subpastas em um nome unico para evitar colisoes
        stem = "_".join(img_path.relative_to(args.src).with_suffix("").parts)
        with Image.open(img_path) as im:
            out_img = process(im, args.resolution, bg)
        out_img.save(args.out / f"{stem}.png")
        shutil.copyfile(caption, args.out / f"{stem}.txt")
        done += 1

    print(f"[OK] {done} imagens processadas em {args.resolution}px -> {args.out}")
    print(f"[INFO] {len(images) - done} ignoradas (sem legenda ou sem processamento)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
