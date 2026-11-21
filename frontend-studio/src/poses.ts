/**
 * Poses do editor: esqueleto COCO-18 em coordenadas normalizadas (x, y em 0..1; y cresce para baixo).
 * Os 14 primeiros pontos sao editaveis; olhos e orelhas (14..17) saem do nariz, como no backend (poses.skeleton).
 */
export type Pt = [number, number];
export type Pose = Pt[];

export const POINT_NAMES = [
  "nariz", "pescoço", "ombro D", "cotovelo D", "mão D", "ombro E", "cotovelo E", "mão E",
  "quadril D", "joelho D", "pé D", "quadril E", "joelho E", "pé E",
] as const;
export const EDITABLE = POINT_NAMES.length; // 14

/** Osso -> [ponto, pai]. O pescoço e a raiz: arrastá-lo move o corpo todo (modo "ossos fixos"). */
export const PARENT: Record<number, number> = {
  0: 1, 2: 1, 5: 1, 3: 2, 4: 3, 6: 5, 7: 6, 8: 1, 11: 1, 9: 8, 10: 9, 12: 11, 13: 12,
};
/** Linhas desenhadas na tela (pares de pontos). */
export const BONES: [number, number][] = [
  [1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
];
/** Cores por membro (D = lado perto da câmera). */
export const BONE_COLORS = [
  "#f4f4f5", "#f97316", "#facc15", "#a3e635", "#38bdf8", "#818cf8", "#e879f9", "#fb7185",
  "#22c55e", "#14b8a6", "#06b6d4", "#3b82f6", "#a855f7",
];
const SWAP: [number, number][] = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13]];

const children = (i: number): number[] => Object.keys(PARENT).map(Number).filter((c) => PARENT[c] === i);
export function descendants(i: number): number[] {
  const out: number[] = [];
  const walk = (n: number) => children(n).forEach((c) => { out.push(c); walk(c); });
  walk(i);
  return out;
}

export const clonePose = (p: Pose): Pose => p.map(([x, y]) => [x, y] as Pt);
export const clamp01 = (v: number, lo = -0.1, hi = 1.1) => Math.min(hi, Math.max(lo, v));

/** 14 pontos editaveis -> 18 pontos (acrescenta olhos e orelhas pela posicao do nariz). */
export function expandPose(p: Pose): Pose {
  const [nx, ny] = p[0];
  const out = clonePose(p.slice(0, EDITABLE));
  out.push([nx - 0.025, ny - 0.012], [nx + 0.005, ny - 0.012], [nx - 0.05, ny + 0.005], [nx - 0.035, ny + 0.005]);
  return out;
}

/**
 * Move o ponto `i` para (x, y). Com `fixedBones`, o osso mantem o comprimento (o ponto fica na direcao do mouse,
 * a distancia original do pai) e os filhos acompanham rigidamente; o pescoco translada o corpo inteiro.
 */
export function movePoint(pose: Pose, i: number, x: number, y: number, fixedBones: boolean): Pose {
  const out = clonePose(pose);
  if (i >= EDITABLE) return out;
  if (!fixedBones) { out[i] = [clamp01(x), clamp01(y)]; return out; }
  let target: Pt = [x, y];
  if (i in PARENT) {
    const [px, py] = pose[PARENT[i]];
    const len = Math.hypot(pose[i][0] - px, pose[i][1] - py);
    const dx = x - px, dy = y - py, d = Math.hypot(dx, dy) || 1;
    target = [px + (dx / d) * len, py + (dy / d) * len];
  }
  const mx = target[0] - pose[i][0], my = target[1] - pose[i][1];
  for (const n of [i, ...descendants(i)]) out[n] = [pose[n][0] + mx, pose[n][1] + my];
  return out;
}

/** Espelho verdadeiro em torno do pescoço: inverte x e troca direita/esquerda. */
export function mirrorPose(pose: Pose): Pose {
  const cx = pose[1][0];
  const out = pose.map(([x, y]) => [2 * cx - x, y] as Pt);
  for (const [a, b] of SWAP) [out[a], out[b]] = [out[b], out[a]];
  return out;
}

/** `n` poses entre a e b (exclusive), com cosseno: começa e termina devagar. */
export function between(a: Pose, b: Pose, n: number): Pose[] {
  const out: Pose[] = [];
  for (let k = 1; k <= n; k++) {
    const t = k / (n + 1), w = 0.5 - 0.5 * Math.cos(Math.PI * t);
    out.push(a.map(([x, y], i) => [x + (b[i][0] - x) * w, y + (b[i][1] - y) * w] as Pt));
  }
  return out;
}

/** Pose em pé, usada quando nenhum movimento pronto foi carregado. */
export const STANDING: Pose = expandPose([
  [0.52, 0.2], [0.5, 0.27], [0.45, 0.285], [0.43, 0.39], [0.44, 0.48], [0.55, 0.285], [0.57, 0.39], [0.56, 0.48],
  [0.48, 0.52], [0.49, 0.7], [0.48, 0.89], [0.52, 0.52], [0.51, 0.7], [0.52, 0.89],
]);

export const isPose = (v: unknown): v is Pose =>
  Array.isArray(v) && v.length === 18 && v.every((p) => Array.isArray(p) && p.length === 2 && p.every((n) => typeof n === "number" && Number.isFinite(n)));
