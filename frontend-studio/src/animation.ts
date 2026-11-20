// Conversões de imagem e exportação (GIF/PNG) do editor de animação. Tudo no navegador, sem dependências.

export async function pngToImageData(b64: string): Promise<ImageData> {
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height);
}

export function imageDataToCanvas(d: ImageData, scale = 1): HTMLCanvasElement {
  const base = document.createElement("canvas");
  base.width = d.width; base.height = d.height;
  (base.getContext("2d") as CanvasRenderingContext2D).putImageData(d, 0, 0);
  if (scale === 1) return base;
  const out = document.createElement("canvas");
  out.width = d.width * scale; out.height = d.height * scale;
  const ctx = out.getContext("2d") as CanvasRenderingContext2D;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(base, 0, 0, out.width, out.height);
  return out;
}

export const canvasToBlob = (c: HTMLCanvasElement): Promise<Blob> =>
  new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("Falha ao gerar o PNG."))), "image/png"));

/** Folha de sprites horizontal (1 linha, frames lado a lado). */
export function spriteSheet(frames: ImageData[], scale = 1): HTMLCanvasElement {
  const w = frames[0].width * scale, h = frames[0].height * scale;
  const out = document.createElement("canvas");
  out.width = w * frames.length; out.height = h;
  const ctx = out.getContext("2d") as CanvasRenderingContext2D;
  ctx.imageSmoothingEnabled = false;
  frames.forEach((f, i) => ctx.drawImage(imageDataToCanvas(f, scale), i * w, 0));
  return out;
}

export function download(blob: Blob, name: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---------------------------------------------------------------- GIF89a

function lzw(indices: Uint8Array, minCode: number): number[] {
  const clear = 1 << minCode, eoi = clear + 1;
  const out: number[] = [];
  let codeSize = minCode + 1, next = eoi + 1, cur = 0, bits = 0;
  const dict = new Map<number, number>();
  const emit = (code: number) => {
    cur |= code << bits; bits += codeSize;
    while (bits >= 8) { out.push(cur & 255); cur >>>= 8; bits -= 8; }
  };
  emit(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i], key = (prefix << 8) | k;
    const found = dict.get(key);
    if (found !== undefined) { prefix = found; continue; }
    emit(prefix);
    if (next === 4096) {
      emit(clear); dict.clear(); codeSize = minCode + 1; next = eoi + 1;
    } else {
      if (next >= (1 << codeSize)) codeSize++;
      dict.set(key, next++);
    }
    prefix = k;
  }
  emit(prefix); emit(eoi);
  if (bits > 0) out.push(cur & 255);
  return out;
}

/**
 * GIF animado em loop. `bg`: cor de fundo "#rrggbb"; null mantém a transparência.
 * Paleta global de 255 cores + índice 0 transparente; excedentes viram a cor mais próxima.
 */
export function encodeGif(frames: ImageData[], fps: number, scale = 1, bg: string | null = null): Blob {
  const src = scale === 1 ? frames : frames.map((f) => {
    const c = imageDataToCanvas(f, scale);
    return (c.getContext("2d") as CanvasRenderingContext2D).getImageData(0, 0, c.width, c.height);
  });
  const w = src[0].width, h = src[0].height;
  const bgRgb = bg ? [1, 3, 5].map((i) => parseInt(bg.slice(i, i + 2), 16)) : null;
  const colorOf = (d: ImageData, i: number): number | null => {
    if (d.data[i + 3] < 128) return bgRgb ? (bgRgb[0] << 16) | (bgRgb[1] << 8) | bgRgb[2] : null;
    return (d.data[i] << 16) | (d.data[i + 1] << 8) | d.data[i + 2];
  };
  const counts = new Map<number, number>();
  for (const d of src) for (let i = 0; i < d.data.length; i += 4) {
    const c = colorOf(d, i);
    if (c !== null) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const palette = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 255).map(([c]) => c);
  const index = new Map<number, number>(palette.map((c, i) => [c, i + 1]));
  const nearest = (c: number): number => {
    const cached = index.get(c);
    if (cached !== undefined) return cached;
    let best = 1, bd = Infinity;
    const r = c >> 16, g = (c >> 8) & 255, b = c & 255;
    palette.forEach((p, i) => {
      const d = (r - (p >> 16)) ** 2 + (g - ((p >> 8) & 255)) ** 2 + (b - (p & 255)) ** 2;
      if (d < bd) { bd = d; best = i + 1; }
    });
    index.set(c, best);
    return best;
  };
  const bytes: number[] = [];
  const u16 = (n: number) => bytes.push(n & 255, (n >> 8) & 255);
  bytes.push(...[..."GIF89a"].map((ch) => ch.charCodeAt(0)));
  u16(w); u16(h); bytes.push(0xf7, 0, 0);
  bytes.push(0, 0, 0);                                            // índice 0 = transparente
  for (let i = 0; i < 255; i++) {
    const c = palette[i] ?? 0;
    bytes.push(c >> 16, (c >> 8) & 255, c & 255);
  }
  bytes.push(0x21, 0xff, 0x0b, ...[..."NETSCAPE2.0"].map((ch) => ch.charCodeAt(0)), 3, 1, 0, 0, 0);
  const delay = Math.max(2, Math.round(100 / fps));
  for (const d of src) {
    const idx = new Uint8Array(w * h);
    for (let p = 0, i = 0; p < idx.length; p++, i += 4) {
      const c = colorOf(d, i);
      idx[p] = c === null ? 0 : nearest(c);
    }
    bytes.push(0x21, 0xf9, 4, bgRgb ? 0x08 : 0x09);              // descarta o frame; bit 0 = transparência
    u16(delay); bytes.push(0, 0);
    bytes.push(0x2c); u16(0); u16(0); u16(w); u16(h); bytes.push(0, 8);
    const data = lzw(idx, 8);
    for (let i = 0; i < data.length; i += 255) {
      const chunk = data.slice(i, i + 255);
      bytes.push(chunk.length, ...chunk);
    }
    bytes.push(0);
  }
  bytes.push(0x3b);
  return new Blob([new Uint8Array(bytes)], { type: "image/gif" });
}
