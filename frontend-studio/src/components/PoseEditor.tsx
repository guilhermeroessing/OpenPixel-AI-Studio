import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft, ChevronRight, Copy, FlipHorizontal, Pause, Play, Redo2, Sparkles, Trash2, Undo2, X,
} from "lucide-react";
import { getMotions, type MotionPreset } from "../api";
import {
  BONES, BONE_COLORS, EDITABLE, POINT_NAMES, STANDING, between, clonePose, expandPose, isPose, mirrorPose, movePoint,
  type Pose,
} from "../poses";

export interface CustomMotionSpec {
  poses: Pose[]; text: string; fps: number; anchor: "fixed" | "centroid"; allowObjects: boolean; extraAvoid: string;
}

const KEY = "openpixel.poses";
const MAX_HISTORY = 100;
const DEFAULT_SPEC: CustomMotionSpec = {
  poses: [clonePose(STANDING), clonePose(STANDING)], text: "", fps: 8, anchor: "fixed", allowObjects: false, extraAvoid: "",
};

function loadDraft(): CustomMotionSpec {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (v && Array.isArray(v.poses) && v.poses.length > 0 && v.poses.every(isPose)) return { ...DEFAULT_SPEC, ...v };
  } catch { /* rascunho indisponível ou inválido */ }
  return { ...DEFAULT_SPEC, poses: DEFAULT_SPEC.poses.map(clonePose) };
}

function drawSkeleton(ctx: CanvasRenderingContext2D, pose: Pose, S: number, opts: {
  alpha?: number; color?: string; point?: number; hover?: number | null;
} = {}) {
  const { alpha = 1, color, point = 6, hover = null } = opts;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.lineCap = "round";
  BONES.forEach(([a, b], k) => {
    ctx.strokeStyle = color ?? BONE_COLORS[k];
    ctx.lineWidth = Math.max(2, point * 0.7);
    ctx.beginPath(); ctx.moveTo(pose[a][0] * S, pose[a][1] * S); ctx.lineTo(pose[b][0] * S, pose[b][1] * S); ctx.stroke();
  });
  for (let i = 0; i < pose.length; i++) {
    const face = i >= EDITABLE;
    ctx.fillStyle = color ?? (face ? "#a1a1aa" : "#fafafa");
    ctx.beginPath();
    ctx.arc(pose[i][0] * S, pose[i][1] * S, face ? point * 0.4 : i === hover ? point * 1.35 : point, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function PoseThumb({ pose, active, onClick, label }: { pose: Pose; active: boolean; onClick: () => void; label: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#18181b"; ctx.fillRect(0, 0, c.width, c.height);
    drawSkeleton(ctx, pose, c.width, { point: 2.2 });
  });
  return (
    <button onClick={onClick} aria-label={`Pose ${label}`} aria-current={active}
      className={`relative shrink-0 rounded border-2 ${active ? "border-emerald-500" : "border-zinc-800 hover:border-zinc-600"}`}>
      <canvas ref={ref} width={64} height={64} className="block h-16 w-16" />
      <span className="absolute bottom-0 right-0 bg-zinc-950/80 px-1 text-[10px] text-zinc-300">{label}</span>
    </button>
  );
}

export default function PoseEditor({ prompt, onClose, onGenerate }: {
  prompt: string; onClose: () => void; onGenerate: (spec: CustomMotionSpec) => void;
}) {
  const draft = useMemo(loadDraft, []);
  const [poses, setPoses] = useState<Pose[]>(draft.poses);
  const [idx, setIdx] = useState(0);
  const [text, setText] = useState(draft.text);
  const [fps, setFps] = useState(draft.fps);
  const [anchor, setAnchor] = useState<"fixed" | "centroid">(draft.anchor);
  const [allowObjects, setAllowObjects] = useState(draft.allowObjects);
  const [extraAvoid, setExtraAvoid] = useState(draft.extraAvoid);
  const [fixedBones, setFixedBones] = useState(true);
  const [onion, setOnion] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [between_, setBetween] = useState(3);
  const [presets, setPresets] = useState<Record<string, MotionPreset> | null>(null);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [size, setSize] = useState(() => Math.max(320, Math.min(640, window.innerHeight - 300)));

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const undoRef = useRef<{ poses: Pose[]; idx: number }[]>([]);
  const redoRef = useRef<{ poses: Pose[]; idx: number }[]>([]);
  const dragging = useRef<number | null>(null);
  const n = poses.length;
  const current = poses[Math.min(idx, n - 1)];

  useEffect(() => {
    getMotions().then(setPresets).catch((e: Error) => setPresetError(e.message));
  }, []);
  useEffect(() => {
    const onResize = () => setSize(Math.max(320, Math.min(640, window.innerHeight - 300)));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  useEffect(() => { // rascunho sobrevive a recarregar a página
    try { localStorage.setItem(KEY, JSON.stringify({ poses, text, fps, anchor, allowObjects, extraAvoid })); } catch { /* sem armazenamento */ }
  }, [poses, text, fps, anchor, allowObjects, extraAvoid]);

  // ---------- histórico
  const snapshot = () => {
    undoRef.current.push({ poses: poses.map(clonePose), idx });
    if (undoRef.current.length > MAX_HISTORY) undoRef.current.shift();
    redoRef.current = [];
  };
  const commit = (next: Pose[], nextIdx = idx) => { snapshot(); setPoses(next); setIdx(Math.max(0, Math.min(nextIdx, next.length - 1))); };
  const undo = () => {
    const s = undoRef.current.pop();
    if (!s) return;
    redoRef.current.push({ poses: poses.map(clonePose), idx });
    setPoses(s.poses); setIdx(s.idx);
  };
  const redo = () => {
    const s = redoRef.current.pop();
    if (!s) return;
    undoRef.current.push({ poses: poses.map(clonePose), idx });
    setPoses(s.poses); setIdx(s.idx);
  };

  // ---------- desenho
  const redraw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d") as CanvasRenderingContext2D;
    const S = c.width;
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = "#0b0b0d"; ctx.fillRect(0, 0, S, S);
    ctx.strokeStyle = "#1f1f23"; ctx.lineWidth = 1;
    for (let g = 1; g < 10; g++) {
      ctx.beginPath(); ctx.moveTo((g / 10) * S, 0); ctx.lineTo((g / 10) * S, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (g / 10) * S); ctx.lineTo(S, (g / 10) * S); ctx.stroke();
    }
    ctx.setLineDash([6, 6]); ctx.strokeStyle = "#3f3f46";
    ctx.beginPath(); ctx.moveTo(0, 0.89 * S); ctx.lineTo(S, 0.89 * S); ctx.stroke(); // chão (onde os pés repousam)
    ctx.setLineDash([]);
    ctx.fillStyle = "#52525b"; ctx.font = "11px sans-serif"; ctx.fillText("chão", 6, 0.89 * S - 4);
    if (onion && !playing) {
      if (idx > 0) drawSkeleton(ctx, expandPose(poses[idx - 1]), S, { alpha: 0.35, color: "#38bdf8", point: 4 });
      if (idx < n - 1) drawSkeleton(ctx, expandPose(poses[idx + 1]), S, { alpha: 0.3, color: "#fb7185", point: 4 });
    }
    drawSkeleton(ctx, expandPose(current), S, { point: 6, hover });
  }, [poses, idx, onion, playing, hover, current, n]);
  useEffect(() => { redraw(); }, [redraw]);

  // prévia animada do esqueleto
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setIdx((i) => (i + 1) % n), 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps, n]);

  // ---------- mouse
  const toNorm = (e: React.PointerEvent): [number, number] => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };
  const nearest = (x: number, y: number): number | null => {
    let best: number | null = null, bd = (14 / size) ** 2;
    for (let i = 0; i < EDITABLE; i++) {
      const d = (current[i][0] - x) ** 2 + (current[i][1] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };
  const onDown = (e: React.PointerEvent) => {
    if (playing) return;
    const [x, y] = toNorm(e);
    const i = nearest(x, y);
    if (i === null) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    snapshot();
    dragging.current = i;
  };
  const onMove = (e: React.PointerEvent) => {
    const [x, y] = toNorm(e);
    const d = dragging.current;
    if (d === null) { const h = nearest(x, y); setHover((p) => (p === h ? p : h)); return; }
    setPoses((ps) => ps.map((p, k) => (k === idx ? movePoint(p, d, x, y, fixedBones) : p)));
  };
  const onUp = () => { dragging.current = null; };

  // ---------- operações de frame
  const addFrame = () => commit([...poses.slice(0, idx + 1), clonePose(current), ...poses.slice(idx + 1)], idx + 1);
  const deleteFrame = () => { if (n > 1) commit(poses.filter((_, k) => k !== idx), idx); };
  const moveFrame = (d: number) => {
    const j = idx + d;
    if (j < 0 || j >= n) return;
    const next = poses.map(clonePose);
    [next[idx], next[j]] = [next[j], next[idx]];
    commit(next, j);
  };
  const mirrorFrame = () => commit(poses.map((p, k) => (k === idx ? mirrorPose(p) : p)));
  const mirrorAll = () => commit(poses.map(mirrorPose));
  const fillBetween = () => {
    if (idx >= n - 1) return;
    commit([...poses.slice(0, idx + 1), ...between(current, poses[idx + 1], between_), ...poses.slice(idx + 1)], idx);
  };
  const applyPreset = (name: string) => {
    const p = presets?.[name];
    if (!p) return;
    commit(p.poses.map((pose) => clonePose(pose as Pose)), 0);
    setFps(p.fps);
    setAnchor(p.anchor === "centroid" ? "centroid" : "fixed");
    setPending(null);
  };
  // substituir poses já editadas pede confirmação na própria tela (sem diálogo do navegador)
  const untouched = n === 2 && poses.every((p) => p.every(([x, y], i) => x === STANDING[i][0] && y === STANDING[i][1]));
  const loadPreset = (name: string) => (untouched ? applyPreset(name) : setPending(name));

  const generate = () => onGenerate({ poses: poses.map(expandPose), text: text.trim(), fps, anchor, allowObjects, extraAvoid: extraAvoid.trim() });

  // ---------- teclado
  const keys = useRef<(e: KeyboardEvent) => void>(() => {});
  keys.current = (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if (k === "arrowleft") setIdx((i) => Math.max(0, i - 1));
    else if (k === "arrowright") setIdx((i) => Math.min(n - 1, i + 1));
    else if (k === " ") { e.preventDefault(); setPlaying((p) => !p); }
    else if (k === "escape") onClose();
  };
  useEffect(() => {
    const fn = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

  const btn = "flex items-center justify-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40";
  const field = "w-full rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-200";

  return (
    <div role="dialog" aria-label="Editor de poses" className="fixed inset-0 z-50 flex flex-col bg-zinc-950 text-zinc-200">
      <header className="flex items-center gap-3 border-b border-zinc-800 px-4 py-2">
        <h2 className="text-sm font-semibold">Editor de poses · animação personalizada</h2>
        <span className="text-xs text-zinc-500">Pose {idx + 1} de {n}</span>
        <div className="ml-auto flex items-center gap-2">
          <button className={btn} onClick={undo} aria-label="Desfazer" title="Desfazer (Ctrl+Z)"><Undo2 size={14} /></button>
          <button className={btn} onClick={redo} aria-label="Refazer" title="Refazer (Ctrl+Shift+Z)"><Redo2 size={14} /></button>
          <button className={btn} onClick={onClose} aria-label="Fechar"><X size={14} /> Fechar</button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col items-center gap-2 overflow-auto p-4">
          <canvas ref={canvasRef} width={size} height={size} aria-label="Esqueleto da pose"
            className={`touch-none rounded border border-zinc-800 ${hover !== null && !playing ? "cursor-grab" : "cursor-default"}`}
            style={{ width: size, height: size }}
            onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
            onPointerLeave={() => setHover(null)} />
          <p className="text-xs text-zinc-500">
            Arraste os pontos brancos. {hover !== null ? <strong className="text-zinc-300">{POINT_NAMES[hover]}</strong> : "Azul = pose anterior, rosa = próxima."}
          </p>
          <div className="flex flex-wrap items-center justify-center gap-1">
            <button className={btn} onClick={() => setPlaying((p) => !p)}>{playing ? <Pause size={12} /> : <Play size={12} />} {playing ? "Parar" : "Prévia"}</button>
            <button className={btn} onClick={() => setIdx((i) => Math.max(0, i - 1))} aria-label="Pose anterior"><ChevronLeft size={12} /></button>
            <button className={btn} onClick={() => setIdx((i) => Math.min(n - 1, i + 1))} aria-label="Próxima pose"><ChevronRight size={12} /></button>
            <button className={btn} onClick={addFrame}><Copy size={12} /> Duplicar pose</button>
            <button className={btn} onClick={deleteFrame} disabled={n <= 1}><Trash2 size={12} /> Excluir</button>
            <button className={btn} onClick={() => moveFrame(-1)} disabled={idx === 0}>← Mover</button>
            <button className={btn} onClick={() => moveFrame(1)} disabled={idx >= n - 1}>Mover →</button>
            <button className={btn} onClick={mirrorFrame}><FlipHorizontal size={12} /> Espelhar pose</button>
            <button className={btn} onClick={mirrorAll}><FlipHorizontal size={12} /> Espelhar todas</button>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <button className={btn} onClick={fillBetween} disabled={idx >= n - 1}>Criar intermediárias entre esta e a próxima</button>
            <input type="number" min={1} max={30} value={between_} aria-label="Quantidade de intermediárias"
              onChange={(e) => setBetween(Math.max(1, Math.min(30, Number(e.target.value) || 1)))} className={`${field} w-14`} />
          </div>
          <div className="flex w-full max-w-4xl gap-2 overflow-x-auto pb-1" aria-label="Poses">
            {poses.map((p, k) => <PoseThumb key={k} pose={expandPose(p)} active={k === idx} label={k + 1} onClick={() => { setPlaying(false); setIdx(k); }} />)}
          </div>
        </main>

        <aside className="w-72 shrink-0 space-y-4 overflow-y-auto border-l border-zinc-800 p-3 text-xs">
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Ponto de partida</h3>
            {presetError && <p className="text-amber-300">Movimentos prontos indisponíveis: {presetError}</p>}
            <div className="flex flex-wrap gap-1">
              {presets && Object.entries(presets).map(([k, p]) => (
                <button key={k} className={btn} onClick={() => loadPreset(k)}>{p.label}</button>
              ))}
              {!presets && !presetError && <span className="text-zinc-500">Carregando…</span>}
            </div>
            {pending && presets?.[pending] && (
              <div role="alert" className="rounded-md border border-amber-700 bg-amber-950/40 p-2">
                <p className="text-amber-200">Substituir as {n} poses atuais por “{presets[pending].label}” ({presets[pending].frames} poses)?</p>
                <div className="mt-1 flex gap-1">
                  <button className={btn} onClick={() => applyPreset(pending)}>Substituir</button>
                  <button className={btn} onClick={() => setPending(null)}>Cancelar</button>
                </div>
              </div>
            )}
          </section>
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Edição</h3>
            <label className="flex items-center justify-between">Manter comprimento dos ossos
              <input type="checkbox" checked={fixedBones} onChange={(e) => setFixedBones(e.target.checked)} className="accent-emerald-500" />
            </label>
            <label className="flex items-center justify-between">Mostrar poses vizinhas
              <input type="checkbox" checked={onion} onChange={(e) => setOnion(e.target.checked)} className="accent-emerald-500" />
            </label>
            <p className="text-zinc-600">Com os ossos fixos, arrastar o cotovelo leva a mão junto e arrastar o pescoço move o corpo todo. Espaço toca a prévia, ← → trocam de pose.</p>
          </section>
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Animação</h3>
            <label className="block">Descrição do movimento (só o gesto do corpo)
              <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} maxLength={300} placeholder="ex.: waving hello with the right hand"
                className={`${field} mt-1`} />
            </label>
            <label className="flex items-center justify-between">Quadros por segundo
              <input type="number" min={1} max={60} value={fps} onChange={(e) => setFps(Math.max(1, Math.min(60, Number(e.target.value) || 1)))} className={`${field} w-16`} />
            </label>
            <label className="block">Posição no quadro
              <select value={anchor} onChange={(e) => setAnchor(e.target.value as "fixed" | "centroid")} className={`${field} mt-1`}>
                <option value="fixed">Local fixo (parado, ataques)</option>
                <option value="centroid">Acompanha o corpo (corrida)</option>
              </select>
            </label>
            <label className="block">Evitar também (opcional)
              <input value={extraAvoid} onChange={(e) => setExtraAvoid(e.target.value)} maxLength={300} className={`${field} mt-1`} placeholder="ex.: hat, glasses" />
            </label>
            <label className="flex items-start gap-2">
              <input type="checkbox" checked={allowObjects} onChange={(e) => setAllowObjects(e.target.checked)} className="mt-0.5 accent-emerald-500" />
              <span>Permitir armas e objetos <span className="text-zinc-500">(por sua conta e risco: o modelo pode inventar itens e deformar as mãos)</span></span>
            </label>
          </section>
          <section className="space-y-2">
            <p className="text-zinc-500">Personagem (barra lateral): <strong className="text-zinc-300">{prompt.trim() || "— escreva um prompt na barra lateral —"}</strong></p>
            <button disabled={!prompt.trim()} onClick={generate}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50">
              <Sparkles size={14} /> Gerar {n} {n === 1 ? "frame" : "frames"} com a IA
            </button>
            <p className="text-zinc-600">São {n} imagens mais 1 de referência do personagem. O tempo depende do seu computador.</p>
          </section>
        </aside>
      </div>
    </div>
  );
}
