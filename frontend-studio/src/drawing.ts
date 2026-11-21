// Operações de desenho sobre ImageData (RGBA, 1 pixel = 1 pixel lógico). Sem DOM: dá para testar à parte.

export type RGBA = [number, number, number, number];

export function cloneData(d: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(d.data), d.width, d.height);
}

export function copyInto(dst: ImageData, src: ImageData): void {
  dst.data.set(src.data);
}

export function getPixel(d: ImageData, x: number, y: number): RGBA | null {
  if (x < 0 || y < 0 || x >= d.width || y >= d.height) return null;
  const i = (y * d.width + x) * 4;
  return [d.data[i], d.data[i + 1], d.data[i + 2], d.data[i + 3]];
}

export function setPixel(d: ImageData, x: number, y: number, c: RGBA): void {
  if (x < 0 || y < 0 || x >= d.width || y >= d.height) return;
  const i = (y * d.width + x) * 4;
  d.data[i] = c[0]; d.data[i + 1] = c[1]; d.data[i + 2] = c[2]; d.data[i + 3] = c[3];
}

/** Pincel quadrado `size` x `size` centrado em (x, y). */
export function stamp(d: ImageData, x: number, y: number, c: RGBA, size = 1): void {
  const off = Math.floor((size - 1) / 2);
  for (let j = 0; j < size; j++) for (let i = 0; i < size; i++) setPixel(d, x - off + i, y - off + j, c);
}

/** Pontos de uma linha de Bresenham (inclui as duas pontas). */
export function linePoints(x0: number, y0: number, x1: number, y1: number): [number, number][] {
  const pts: [number, number][] = [];
  const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  let err = dx + dy, x = x0, y = y0;
  for (;;) {
    pts.push([x, y]);
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
  return pts;
}

export function drawLine(d: ImageData, x0: number, y0: number, x1: number, y1: number, c: RGBA, size = 1): void {
  for (const [x, y] of linePoints(x0, y0, x1, y1)) stamp(d, x, y, c, size);
}

export function drawRect(d: ImageData, x0: number, y0: number, x1: number, y1: number, c: RGBA, size = 1, filled = false): void {
  const [ax, bx] = [Math.min(x0, x1), Math.max(x0, x1)];
  const [ay, by] = [Math.min(y0, y1), Math.max(y0, y1)];
  if (filled) {
    for (let y = ay; y <= by; y++) for (let x = ax; x <= bx; x++) setPixel(d, x, y, c);
    return;
  }
  drawLine(d, ax, ay, bx, ay, c, size); drawLine(d, ax, by, bx, by, c, size);
  drawLine(d, ax, ay, ax, by, c, size); drawLine(d, bx, ay, bx, by, c, size);
}

export function drawEllipse(d: ImageData, x0: number, y0: number, x1: number, y1: number, c: RGBA, size = 1, filled = false): void {
  const [ax, bx] = [Math.min(x0, x1), Math.max(x0, x1)];
  const [ay, by] = [Math.min(y0, y1), Math.max(y0, y1)];
  const cx = (ax + bx) / 2, cy = (ay + by) / 2, rx = (bx - ax) / 2, ry = (by - ay) / 2;
  if (filled) {
    for (let y = ay; y <= by; y++) {
      const t = ry === 0 ? 0 : (y - cy) / ry;
      if (Math.abs(t) > 1) continue;
      const half = rx * Math.sqrt(1 - t * t);
      for (let x = Math.round(cx - half); x <= Math.round(cx + half); x++) setPixel(d, x, y, c);
    }
    return;
  }
  const n = Math.max(16, Math.ceil(Math.PI * 2 * Math.max(rx, ry) * 2));
  let prev: [number, number] | null = null;
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2;
    const p: [number, number] = [Math.round(cx + rx * Math.cos(a)), Math.round(cy + ry * Math.sin(a))];
    if (prev) drawLine(d, prev[0], prev[1], p[0], p[1], c, size); else stamp(d, p[0], p[1], c, size);
    prev = p;
  }
}

/** Balde de tinta: troca a região contígua (4 vizinhos) da mesma cor exata. Devolve quantos pixels mudou. */
export function floodFill(d: ImageData, x: number, y: number, c: RGBA): number {
  const start = getPixel(d, x, y);
  if (!start) return 0;
  if (start[0] === c[0] && start[1] === c[1] && start[2] === c[2] && start[3] === c[3]) return 0;
  const { width: w, height: h, data } = d;
  const same = (i: number) => data[i] === start[0] && data[i + 1] === start[1] && data[i + 2] === start[2] && data[i + 3] === start[3];
  const stack: number[] = [x, y];
  let count = 0;
  while (stack.length) {
    const cy = stack.pop() as number, cx = stack.pop() as number;
    let lx = cx;
    while (lx >= 0 && same((cy * w + lx) * 4)) lx--;
    lx++;
    let up = false, down = false;
    for (let i = lx; i < w && same((cy * w + i) * 4); i++) {
      const o = (cy * w + i) * 4;
      data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = c[3];
      count++;
      if (cy > 0) {
        const s = same(((cy - 1) * w + i) * 4);
        if (s && !up) { stack.push(i, cy - 1); up = true; } else if (!s) up = false;
      }
      if (cy < h - 1) {
        const s = same(((cy + 1) * w + i) * 4);
        if (s && !down) { stack.push(i, cy + 1); down = true; } else if (!s) down = false;
      }
    }
  }
  return count;
}

/** Desloca o conteúdo (dx, dy) a partir de `src`, escrevendo em `dst`; o que sai da tela é cortado. */
export function translate(dst: ImageData, src: ImageData, dx: number, dy: number): void {
  dst.data.fill(0);
  const { width: w, height: h } = src;
  for (let y = 0; y < h; y++) {
    const ty = y + dy;
    if (ty < 0 || ty >= h) continue;
    for (let x = 0; x < w; x++) {
      const tx = x + dx;
      if (tx < 0 || tx >= w) continue;
      const s = (y * w + x) * 4, t = (ty * w + tx) * 4;
      dst.data[t] = src.data[s]; dst.data[t + 1] = src.data[s + 1]; dst.data[t + 2] = src.data[s + 2]; dst.data[t + 3] = src.data[s + 3];
    }
  }
}

export function flipHorizontal(d: ImageData): void {
  const src = cloneData(d), w = d.width;
  for (let y = 0; y < d.height; y++) {
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * 4, t = (y * w + (w - 1 - x)) * 4;
      d.data[t] = src.data[s]; d.data[t + 1] = src.data[s + 1]; d.data[t + 2] = src.data[s + 2]; d.data[t + 3] = src.data[s + 3];
    }
  }
}

export const hexToRgba = (hex: string, a = 255): RGBA => {
  const v = hex.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16), a];
};

export const rgbaToHex = (c: RGBA | number[]): string =>
  "#" + [c[0], c[1], c[2]].map((n) => n.toString(16).padStart(2, "0")).join("");

/** Cores opacas distintas (ordenadas por frequência) de vários frames: alimenta a paleta do editor. */
export function collectPalette(frames: ImageData[], limit = 64): string[] {
  const counts = new Map<number, number>();
  for (const f of frames) {
    for (let i = 0; i < f.data.length; i += 4) {
      if (f.data[i + 3] < 128) continue;
      const k = (f.data[i] << 16) | (f.data[i + 1] << 8) | f.data[i + 2];
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit)
    .map(([k]) => "#" + k.toString(16).padStart(6, "0"));
}

/**
 * Varinha de remover fundo: apaga (alfa 0) os pixels parecidos com o pixel clicado. `tol` e a soma das diferencas
 * RGB (0 = so a cor exata). Contiguo: so a regiao ligada ao clique (4 vizinhos); senao, todos os pixels parecidos do
 * frame. Devolve quantos pixels apagou.
 */
export function eraseSimilar(d: ImageData, x: number, y: number, tol: number, contiguous = true): number {
  const start = getPixel(d, x, y);
  if (!start || start[3] === 0) return 0;
  const { width: w, height: h, data } = d;
  const like = (i: number) =>
    data[i + 3] > 0 && Math.abs(data[i] - start[0]) + Math.abs(data[i + 1] - start[1]) + Math.abs(data[i + 2] - start[2]) <= tol;
  let count = 0;
  if (!contiguous) {
    for (let i = 0; i < w * h * 4; i += 4) if (like(i)) { data[i + 3] = 0; count++; }
    return count;
  }
  const seen = new Uint8Array(w * h);
  const stack = [y * w + x];
  seen[y * w + x] = 1;
  while (stack.length) {
    const p = stack.pop() as number;
    if (!like(p * 4)) continue;
    data[p * 4 + 3] = 0;
    count++;
    const px = p % w, py = (p - px) / w;
    if (px > 0 && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
    if (px < w - 1 && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
    if (py > 0 && !seen[p - w]) { seen[p - w] = 1; stack.push(p - w); }
    if (py < h - 1 && !seen[p + w]) { seen[p + w] = 1; stack.push(p + w); }
  }
  return count;
}
