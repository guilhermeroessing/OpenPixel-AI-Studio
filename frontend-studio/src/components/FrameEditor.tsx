import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Check, ChevronLeft, ChevronRight, Circle, Copy, Download, Eraser, FlipHorizontal, Grid3x3, Layers, Minus,
  Move, PaintBucket, Pause, Pencil, Pipette, Play, Plus, Redo2, Square, Trash2, Undo2, Wand2, X, ZoomIn, ZoomOut,
} from "lucide-react";
import {
  type RGBA, cloneData, collectPalette, copyInto, drawEllipse, drawLine, drawRect, eraseSimilar, flipHorizontal, floodFill,
  getPixel, hexToRgba, rgbaToHex, stamp, translate,
} from "../drawing";
import { canvasToBlob, download, encodeGif, imageDataToCanvas, spriteSheet } from "../animation";

type Tool = "pencil" | "eraser" | "line" | "rect" | "ellipse" | "fill" | "picker" | "move" | "wand";

const TOOLS: { id: Tool; label: string; key: string; Icon: typeof Pencil }[] = [
  { id: "pencil", label: "Lápis", key: "B", Icon: Pencil },
  { id: "eraser", label: "Borracha", key: "E", Icon: Eraser },
  { id: "line", label: "Linha", key: "L", Icon: Minus },
  { id: "rect", label: "Retângulo", key: "R", Icon: Square },
  { id: "ellipse", label: "Elipse", key: "O", Icon: Circle },
  { id: "fill", label: "Balde", key: "G", Icon: PaintBucket },
  { id: "picker", label: "Conta-gotas", key: "I", Icon: Pipette },
  { id: "move", label: "Mover frame", key: "M", Icon: Move },
  { id: "wand", label: "Varinha (remover fundo)", key: "W", Icon: Wand2 },
];
const CLEAR: RGBA = [0, 0, 0, 0];
const MAX_UNDO = 60;

function Thumb({ data, active, onClick, label }: { data: ImageData; active: boolean; onClick: () => void; label: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (c) (c.getContext("2d") as CanvasRenderingContext2D).putImageData(data, 0, 0);
  });
  return (
    <button onClick={onClick} aria-label={`Frame ${label}`} aria-current={active}
      className={`relative shrink-0 rounded border-2 ${active ? "border-emerald-500" : "border-zinc-800 hover:border-zinc-600"} checker`}>
      <canvas ref={ref} width={data.width} height={data.height} className="pixelated block" style={{ width: 64, height: 64 }} />
      <span className="absolute bottom-0 right-0 bg-zinc-950/80 px-1 text-[10px] text-zinc-300">{label}</span>
    </button>
  );
}

export default function FrameEditor({ frames: initial, fps: initialFps, onApply, onClose }: {
  frames: ImageData[]; fps: number; onApply: (frames: ImageData[], fps: number) => void; onClose: () => void;
}) {
  const w = initial[0].width, h = initial[0].height;
  const framesRef = useRef<ImageData[]>(initial.map(cloneData));
  const undoRef = useRef<ImageData[][]>(initial.map(() => []));
  const redoRef = useRef<ImageData[][]>(initial.map(() => []));
  const [, setVersion] = useState(0);
  const bump = () => setVersion((v) => v + 1);
  const [idx, setIdx] = useState(0);
  const [tool, setTool] = useState<Tool>("pencil");
  const [color, setColor] = useState("#e43b44");
  const [brush, setBrush] = useState(1);
  const [filled, setFilled] = useState(false);
  const [wandTol, setWandTol] = useState(40);
  const [wandAll, setWandAll] = useState(false);
  const [wandContig, setWandContig] = useState(true);
  const [grid, setGrid] = useState(true);
  const [onion, setOnion] = useState(true);
  const [zoom, setZoom] = useState(() => Math.max(1, Math.min(24, Math.floor(Math.min((window.innerWidth - 600) / w, (window.innerHeight - 330) / h)))));
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(initialFps);
  const [dirty, setDirty] = useState(false);
  const [exportScale, setExportScale] = useState(4);
  const [exportBg, setExportBg] = useState<string | null>(null);
  const [recent, setRecent] = useState<string[]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);

  const mainRef = useRef<HTMLCanvasElement>(null);
  const onionRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ tool: Tool; start: [number, number]; last: [number, number]; snap: ImageData | null; erase: boolean } | null>(null);
  const hoverRef = useRef<[number, number] | null>(null);
  const n = framesRef.current.length;
  const frame = framesRef.current[Math.min(idx, n - 1)];
  const palette = useMemo(() => collectPalette(framesRef.current), [initial]); // paleta original dos frames gerados

  // ---------- histórico
  const pushUndo = (i: number) => {
    const u = undoRef.current[i];
    u.push(cloneData(framesRef.current[i]));
    if (u.length > MAX_UNDO) u.shift();
    redoRef.current[i] = [];
    setDirty(true);
  };
  const undo = () => {
    const u = undoRef.current[idx];
    if (!u.length) return;
    redoRef.current[idx].push(cloneData(framesRef.current[idx]));
    copyInto(framesRef.current[idx], u.pop() as ImageData);
    bump();
  };
  const redo = () => {
    const r = redoRef.current[idx];
    if (!r.length) return;
    undoRef.current[idx].push(cloneData(framesRef.current[idx]));
    copyInto(framesRef.current[idx], r.pop() as ImageData);
    bump();
  };

  // ---------- desenho imperativo (sem re-render a cada movimento do mouse)
  const col = (): RGBA => hexToRgba(color);
  const redraw = () => {
    const f = framesRef.current[Math.min(idx, framesRef.current.length - 1)];
    const main = mainRef.current, on = onionRef.current, ov = overlayRef.current;
    if (!main || !on || !ov) return;
    (main.getContext("2d") as CanvasRenderingContext2D).putImageData(f, 0, 0);
    const octx = on.getContext("2d") as CanvasRenderingContext2D;
    octx.clearRect(0, 0, w, h);
    const count = framesRef.current.length;
    if (onion && count > 1) {
      const ghost = (src: ImageData, tint: string) => {
        const c = imageDataToCanvas(src);
        const cx = c.getContext("2d") as CanvasRenderingContext2D;
        cx.globalCompositeOperation = "source-atop"; cx.fillStyle = tint; cx.fillRect(0, 0, w, h);
        octx.globalAlpha = 0.35; octx.drawImage(c, 0, 0); octx.globalAlpha = 1;
      };
      ghost(framesRef.current[(idx - 1 + count) % count], "#ff4d4d");
      ghost(framesRef.current[(idx + 1) % count], "#4d7bff");
    }
    const ctx = ov.getContext("2d") as CanvasRenderingContext2D;
    ctx.clearRect(0, 0, ov.width, ov.height);
    ctx.imageSmoothingEnabled = false;
    const d = drag.current;
    if (d && (d.tool === "line" || d.tool === "rect" || d.tool === "ellipse") && hoverRef.current) {
      const scratch = new ImageData(w, h);
      const [x1, y1] = hoverRef.current, c: RGBA = [...hexToRgba(color).slice(0, 3), 200] as RGBA;
      if (d.tool === "line") drawLine(scratch, d.start[0], d.start[1], x1, y1, c, brush);
      else if (d.tool === "rect") drawRect(scratch, d.start[0], d.start[1], x1, y1, c, brush, filled);
      else drawEllipse(scratch, d.start[0], d.start[1], x1, y1, c, brush, filled);
      ctx.drawImage(imageDataToCanvas(scratch), 0, 0, ov.width, ov.height);
    }
    if (grid && zoom >= 6) {
      ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let x = 0; x <= w; x++) { ctx.moveTo(x * zoom + 0.5, 0); ctx.lineTo(x * zoom + 0.5, h * zoom); }
      for (let y = 0; y <= h; y++) { ctx.moveTo(0, y * zoom + 0.5); ctx.lineTo(w * zoom, y * zoom + 0.5); }
      ctx.stroke();
    }
    const hv = hoverRef.current;
    if (hv && (tool === "pencil" || tool === "eraser")) {
      const off = Math.floor((brush - 1) / 2);
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1;
      ctx.strokeRect((hv[0] - off) * zoom + 0.5, (hv[1] - off) * zoom + 0.5, brush * zoom - 1, brush * zoom - 1);
    }
  };
  useLayoutEffect(redraw);

  const pos = (e: React.PointerEvent): [number, number] => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return [Math.floor((e.clientX - r.left) / zoom), Math.floor((e.clientY - r.top) / zoom)];
  };
  const remember = (hex: string) => setRecent((r) => [hex, ...r.filter((c) => c !== hex)].slice(0, 12));

  const onDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setPlaying(false);
    const [x, y] = pos(e);
    const erase = e.button === 2;
    const t: Tool = erase && (tool === "pencil" || tool === "line" || tool === "rect" || tool === "ellipse") ? "eraser" : tool;
    drag.current = { tool: t, start: [x, y], last: [x, y], snap: null, erase };
    hoverRef.current = [x, y];
    const f = frame;
    if (t === "pencil" || t === "eraser") { pushUndo(idx); stamp(f, x, y, t === "eraser" ? CLEAR : col(), brush); remember(color); }
    else if (t === "fill") {
      pushUndo(idx);
      if (floodFill(f, x, y, erase ? CLEAR : col()) === 0) undoRef.current[idx].pop();
      remember(color); bump();
    } else if (t === "wand") {
      // apaga a regiao parecida com o pixel clicado; com "todos os frames" repete no mesmo ponto de cada frame
      const targets = wandAll ? framesRef.current.map((_, i) => i) : [idx];
      let total = 0;
      for (const i of targets) {
        pushUndo(i);
        const c = eraseSimilar(framesRef.current[i], x, y, wandTol, wandContig);
        if (c === 0) undoRef.current[i].pop(); else total += c;
      }
      if (total > 0) bump();
    } else if (t === "picker") {
      const p = getPixel(f, x, y);
      if (p && p[3] > 0) setColor(rgbaToHex(p));
    } else if (t === "move") { pushUndo(idx); drag.current.snap = cloneData(f); }
    redraw();
  };
  const onMove = (e: React.PointerEvent) => {
    const [x, y] = pos(e);
    hoverRef.current = [x, y];
    const d = drag.current;
    setHover((p) => (p && p[0] === x && p[1] === y ? p : [x, y]));
    if (!d) { redraw(); return; }
    const f = frame;
    if (d.tool === "pencil" || d.tool === "eraser") {
      drawLine(f, d.last[0], d.last[1], x, y, d.tool === "eraser" ? CLEAR : col(), brush);
      d.last = [x, y];
    } else if (d.tool === "move" && d.snap) {
      translate(f, d.snap, x - d.start[0], y - d.start[1]);
    }
    redraw();
  };
  const onUp = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const [x, y] = pos(e);
    const f = frame;
    if (d.tool === "line" || d.tool === "rect" || d.tool === "ellipse") {
      pushUndo(idx);
      const c = d.erase ? CLEAR : col();
      if (d.tool === "line") drawLine(f, d.start[0], d.start[1], x, y, c, brush);
      else if (d.tool === "rect") drawRect(f, d.start[0], d.start[1], x, y, c, brush, filled);
      else drawEllipse(f, d.start[0], d.start[1], x, y, c, brush, filled);
      remember(color);
    }
    drag.current = null;
    bump();
  };

  // ---------- operações de frame
  const spliceAll = (start: number, del: number, ins: ImageData[]) => {
    framesRef.current.splice(start, del, ...ins);
    undoRef.current.splice(start, del, ...ins.map((): ImageData[] => []));
    redoRef.current.splice(start, del, ...ins.map((): ImageData[] => []));
    setDirty(true);
    bump(); // o índice pode não mudar (ex.: excluir o último frame): força a atualização da tela
  };
  const addFrame = (clone: boolean) => {
    spliceAll(idx + 1, 0, [clone ? cloneData(frame) : new ImageData(w, h)]);
    setIdx(idx + 1);
  };
  const deleteFrame = () => {
    if (n <= 1) return;
    spliceAll(idx, 1, []);
    setIdx(Math.min(idx, n - 2));
  };
  const swap = (a: number, b: number) => {
    if (b < 0 || b >= n) return;
    const f = framesRef.current, u = undoRef.current, r = redoRef.current;
    [f[a], f[b]] = [f[b], f[a]]; [u[a], u[b]] = [u[b], u[a]]; [r[a], r[b]] = [r[b], r[a]];
    setIdx(b); setDirty(true); bump();
  };
  const copyPrev = () => {
    if (n < 2) return;
    pushUndo(idx);
    copyInto(frame, framesRef.current[(idx - 1 + n) % n]);
    bump();
  };
  const clearFrame = () => { pushUndo(idx); frame.data.fill(0); bump(); };
  const flip = () => { pushUndo(idx); flipHorizontal(frame); bump(); };

  // ---------- reprodução
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setIdx((i) => (i + 1) % framesRef.current.length), 1000 / Math.max(1, fps));
    return () => clearInterval(id);
  }, [playing, fps]);

  // ---------- atalhos (handlers sempre atuais via ref)
  const keys = useRef<(e: KeyboardEvent) => void>(() => undefined);
  keys.current = (e) => {
    const tag = (e.target as HTMLElement).tagName;
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); redo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = TOOLS.find((x) => x.key.toLowerCase() === k);
    if (t) { setTool(t.id); return; }
    if (k === "arrowleft") { setPlaying(false); setIdx((i) => (i - 1 + n) % n); }
    else if (k === "arrowright") { setPlaying(false); setIdx((i) => (i + 1) % n); }
    else if (k === " ") { e.preventDefault(); setPlaying((p) => !p); }
    else if (k === "[") setBrush((b) => Math.max(1, b - 1));
    else if (k === "]") setBrush((b) => Math.min(16, b + 1));
    else if (k === "h") setGrid((g) => !g);
    else if (k === "n") setOnion((o) => !o);
    else if (k === "+" || k === "=") setZoom((z) => Math.min(32, z + 1));
    else if (k === "-") setZoom((z) => Math.max(1, z - 1));
    else if (k === "escape") close();
  };
  useEffect(() => {
    const fn = (e: KeyboardEvent) => keys.current(e);
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, []);

  const close = () => { if (!dirty || window.confirm("Descartar as alterações feitas nos frames?")) onClose(); };
  const apply = () => onApply(framesRef.current.map(cloneData), fps);

  const exportGif = () => download(encodeGif(framesRef.current, fps, exportScale, exportBg), "corrida.gif");
  const exportSheet = async () => download(await canvasToBlob(spriteSheet(framesRef.current, exportScale)), "corrida_folha.png");
  const exportFrame = async () => download(await canvasToBlob(imageDataToCanvas(frame, exportScale)), `frame_${idx + 1}.png`);

  const btn = "flex items-center justify-center gap-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40";
  const toolBtn = (active: boolean) =>
    `flex h-9 w-9 items-center justify-center rounded-md border ${active ? "border-emerald-500 bg-emerald-600/20 text-emerald-300" : "border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"}`;

  return (
    <div role="dialog" aria-label="Editor de frames" className="fixed inset-0 z-50 flex flex-col bg-zinc-950 text-zinc-200">
      <header className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
        <h2 className="mr-2 text-sm font-semibold">Correção manual · frame {Math.min(idx, n - 1) + 1}/{n}</h2>
        <button className={btn} onClick={undo} aria-label="Desfazer" title="Desfazer (Ctrl+Z)"><Undo2 size={14} /></button>
        <button className={btn} onClick={redo} aria-label="Refazer" title="Refazer (Ctrl+Shift+Z)"><Redo2 size={14} /></button>
        <button className={btn} onClick={() => setZoom((z) => Math.max(1, z - 1))} aria-label="Diminuir zoom"><ZoomOut size={14} /></button>
        <span className="w-10 text-center text-xs text-zinc-400">{zoom}x</span>
        <button className={btn} onClick={() => setZoom((z) => Math.min(32, z + 1))} aria-label="Aumentar zoom"><ZoomIn size={14} /></button>
        <button className={`${btn} ${grid ? "border-emerald-700 text-emerald-300" : ""}`} onClick={() => setGrid((g) => !g)} title="Grade (H)"><Grid3x3 size={14} /> Grade</button>
        <button className={`${btn} ${onion ? "border-emerald-700 text-emerald-300" : ""}`} onClick={() => setOnion((o) => !o)} title="Fantasma dos frames vizinhos (N): anterior em vermelho, próximo em azul"><Layers size={14} /> Onion skin</button>
        <div className="ml-auto flex items-center gap-2">
          <button className={btn} onClick={close}><X size={14} /> Cancelar</button>
          <button className="flex items-center gap-1 rounded-md bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-500" onClick={apply}>
            <Check size={14} /> Aplicar e voltar
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <nav className="flex w-14 flex-col items-center gap-2 border-r border-zinc-800 p-2" aria-label="Ferramentas">
          {TOOLS.map(({ id, label, key, Icon }) => (
            <button key={id} className={toolBtn(tool === id)} aria-label={label} aria-pressed={tool === id} title={`${label} (${key})`} onClick={() => setTool(id)}>
              <Icon size={16} />
            </button>
          ))}
        </nav>

        <main className="relative flex min-w-0 flex-1 items-center justify-center overflow-auto bg-zinc-900 p-4">
          <div className="checker relative shrink-0 shadow-2xl" style={{ width: w * zoom, height: h * zoom }}>
            <canvas ref={onionRef} width={w} height={h} className="pixelated absolute left-0 top-0" style={{ width: w * zoom, height: h * zoom }} />
            <canvas ref={mainRef} width={w} height={h} aria-label="Frame atual" className="pixelated absolute left-0 top-0" style={{ width: w * zoom, height: h * zoom }} />
            <canvas ref={overlayRef} width={w * zoom} height={h * zoom} aria-label="Área de desenho"
              className="absolute left-0 top-0 cursor-crosshair touch-none"
              onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
              onPointerLeave={() => { hoverRef.current = null; setHover(null); if (!drag.current) redraw(); }}
              onContextMenu={(e) => e.preventDefault()} />
          </div>
        </main>

        <aside className="w-64 shrink-0 space-y-4 overflow-y-auto border-l border-zinc-800 p-3 text-xs">
          <section>
            <h3 className="mb-1 font-semibold text-zinc-400">Cor</h3>
            <div className="flex items-center gap-2">
              <input type="color" value={color} onChange={(e) => setColor(e.target.value)} aria-label="Cor atual" className="h-8 w-10 cursor-pointer rounded border border-zinc-700 bg-transparent" />
              <span className="font-mono text-zinc-300">{color}</span>
            </div>
            {recent.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1" aria-label="Cores recentes">
                {recent.map((c) => <button key={c} onClick={() => setColor(c)} title={c} className="h-5 w-5 rounded border border-zinc-700" style={{ background: c }} />)}
              </div>
            )}
          </section>
          <section>
            <h3 className="mb-1 font-semibold text-zinc-400">Paleta do personagem</h3>
            <div className="flex flex-wrap gap-1" aria-label="Paleta">
              {palette.map((c) => (
                <button key={c} onClick={() => setColor(c)} title={c}
                  className={`h-5 w-5 rounded border ${c === color ? "border-white" : "border-zinc-700"}`} style={{ background: c }} />
              ))}
            </div>
          </section>
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Pincel</h3>
            <label className="flex items-center justify-between">Tamanho {brush}px
              <input type="range" min={1} max={16} value={brush} onChange={(e) => setBrush(Number(e.target.value))} className="w-28 accent-emerald-500" />
            </label>
            <label className="flex items-center justify-between">Formas preenchidas
              <input type="checkbox" checked={filled} onChange={(e) => setFilled(e.target.checked)} className="accent-emerald-500" />
            </label>
            <p className="text-zinc-600">Botão direito apaga. Espaço reproduz, ← → trocam de frame, [ ] mudam o pincel.</p>
          </section>
          {tool === "wand" && (
            <section className="space-y-2" aria-label="Varinha">
              <h3 className="font-semibold text-zinc-400">Varinha (remover fundo)</h3>
              <label className="flex items-center justify-between">Tolerância {wandTol}
                <input type="range" min={0} max={200} value={wandTol} onChange={(e) => setWandTol(Number(e.target.value))} className="w-28 accent-emerald-500" />
              </label>
              <label className="flex items-center justify-between">Só região ligada
                <input type="checkbox" checked={wandContig} onChange={(e) => setWandContig(e.target.checked)} className="accent-emerald-500" />
              </label>
              <label className="flex items-center justify-between">Em todos os frames
                <input type="checkbox" checked={wandAll} onChange={(e) => setWandAll(e.target.checked)} className="accent-emerald-500" />
              </label>
              <p className="text-zinc-600">Clique numa cor do fundo para apagar as áreas parecidas. Desfazer (Ctrl+Z) vale por frame.</p>
            </section>
          )}
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Este frame</h3>
            <div className="grid grid-cols-2 gap-1">
              <button className={btn} onClick={copyPrev} disabled={n < 2}><Copy size={12} /> Copiar anterior</button>
              <button className={btn} onClick={flip}><FlipHorizontal size={12} /> Espelhar</button>
              <button className={btn} onClick={clearFrame}><Eraser size={12} /> Limpar</button>
              <button className={btn} onClick={deleteFrame} disabled={n <= 1}><Trash2 size={12} /> Excluir</button>
            </div>
          </section>
          <section className="space-y-2">
            <h3 className="font-semibold text-zinc-400">Exportar</h3>
            <div className="flex items-center justify-between">
              <label htmlFor="exp-scale">Escala</label>
              <select id="exp-scale" value={exportScale} onChange={(e) => setExportScale(Number(e.target.value))} className="rounded bg-zinc-900 border border-zinc-700 px-1 py-0.5">
                {[1, 2, 4, 8].map((s) => <option key={s} value={s}>{s}x</option>)}
              </select>
            </div>
            <div className="flex items-center justify-between">
              <label htmlFor="exp-bg">Fundo do GIF</label>
              <span className="flex items-center gap-1">
                <input id="exp-bg" type="checkbox" checked={exportBg !== null} onChange={(e) => setExportBg(e.target.checked ? "#94a3b8" : null)} className="accent-emerald-500" aria-label="Usar cor de fundo" />
                {exportBg !== null
                  ? <input type="color" value={exportBg} onChange={(e) => setExportBg(e.target.value)} className="h-5 w-8" aria-label="Cor do fundo do GIF" />
                  : <span className="text-zinc-500">transparente</span>}
              </span>
            </div>
            <div className="grid grid-cols-1 gap-1">
              <button className={btn} onClick={exportGif}><Download size={12} /> GIF animado</button>
              <button className={btn} onClick={exportSheet}><Download size={12} /> Folha de sprites (PNG)</button>
              <button className={btn} onClick={exportFrame}><Download size={12} /> Frame atual (PNG)</button>
            </div>
          </section>
        </aside>
      </div>

      <footer className="border-t border-zinc-800 p-2">
        <div className="mb-2 flex items-center gap-2 text-xs">
          <button className={btn} onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pausar" : "Reproduzir"}>
            {playing ? <Pause size={14} /> : <Play size={14} />} {playing ? "Pausar" : "Reproduzir"}
          </button>
          <label className="flex items-center gap-1">FPS
            <input type="number" min={1} max={30} value={fps} aria-label="Quadros por segundo"
              onChange={(e) => setFps(Math.min(30, Math.max(1, Number(e.target.value) || 1)))}
              className="w-14 rounded bg-zinc-900 border border-zinc-700 px-1 py-0.5 text-right" />
          </label>
          <button className={btn} onClick={() => swap(idx, idx - 1)} disabled={idx === 0} aria-label="Mover frame para a esquerda"><ChevronLeft size={14} /></button>
          <button className={btn} onClick={() => swap(idx, idx + 1)} disabled={idx >= n - 1} aria-label="Mover frame para a direita"><ChevronRight size={14} /></button>
          <button className={btn} onClick={() => addFrame(true)}><Copy size={12} /> Duplicar</button>
          <button className={btn} onClick={() => addFrame(false)}><Plus size={12} /> Frame vazio</button>
          <span className="ml-auto text-zinc-500">
            {w}x{h}px · {tool}{hover ? ` · ${hover[0]},${hover[1]}` : ""}
          </span>
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1" aria-label="Frames">
          {framesRef.current.map((f, i) => <Thumb key={i} data={f} active={i === idx} label={i + 1} onClick={() => { setPlaying(false); setIdx(i); }} />)}
        </div>
      </footer>
    </div>
  );
}
