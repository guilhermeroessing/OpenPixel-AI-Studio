import { useEffect, useRef, useState } from "react";
import { Film, Footprints, Loader2, Map as MapIcon, Pencil, PencilRuler, PersonStanding } from "lucide-react";
import {
  generate8, generateMap, getJob, startAnimation, toDataUrl,
  type AnimationResponse, type CompleteMapResponse, type GenOverrides, type SpriteResponse,
} from "../api";
import { imageDataToCanvas, pngToImageData } from "../animation";
import type { CustomMotionSpec } from "./PoseEditor";

export interface AnimState { frames: ImageData[]; fps: number }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function AnimPreview({ anim }: { anim: AnimState }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let i = 0;
    const draw = () => {
      const c = ref.current;
      if (!c) return;
      const ctx = c.getContext("2d") as CanvasRenderingContext2D;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(imageDataToCanvas(anim.frames[i % anim.frames.length]), 0, 0, c.width, c.height);
      i++;
    };
    draw();
    const id = setInterval(draw, 1000 / Math.max(1, anim.fps));
    return () => clearInterval(id);
  }, [anim]);
  const f = anim.frames[0];
  return <canvas ref={ref} width={f.width} height={f.height} aria-label="Prévia da animação"
    className="pixelated checker h-28 w-28 shrink-0 rounded border border-zinc-800" />;
}

type Tab = "animate" | "edit" | "map";

const isSprite = (v: unknown): v is SpriteResponse => !!v && typeof v === "object" && "image_b64" in v;

// /generate8 pode devolver lista ou dicionário (direção -> sprite); normaliza.
function normalize8(data: unknown): [string, SpriteResponse][] {
  const raw: unknown = data && typeof data === "object" && "directions" in data
    ? (data as { directions: unknown }).directions : data;
  if (Array.isArray(raw)) return raw.filter(isSprite).map((s, i) => [String(i + 1), s]);
  if (raw && typeof raw === "object") {
    return Object.entries(raw as Record<string, unknown>).filter((e): e is [string, SpriteResponse] => isSprite(e[1]));
  }
  return [];
}

export default function FloatingPanel({ prompt, size, sprite, overrides, anim, setAnim, openEditor, openPoseEditor, custom, onCustomStarted }: {
  prompt: string; size: number; sprite: SpriteResponse | null; overrides: GenOverrides;
  anim: AnimState | null; setAnim: (a: AnimState) => void; openEditor: () => void;
  openPoseEditor: () => void; custom: CustomMotionSpec | null; onCustomStarted: () => void;
}) {
  const [tab, setTab] = useState<Tab>("animate");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirs, setDirs] = useState<[string, SpriteResponse][]>([]);
  const [map, setMap] = useState<CompleteMapResponse | null>(null);
  const [job, setJob] = useState<{ progress: number; total: number; elapsed: number } | null>(null);

  async function run(fn: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) { setError(e instanceof Error ? e.message : "Erro desconhecido."); }
    finally { setBusy(false); }
  }
  const need = () => { if (!prompt.trim()) throw new Error("Escreva um prompt na barra lateral primeiro."); };

  async function runAnimation(spec?: CustomMotionSpec) {
    need();
    const base = { prompt, size, kind: "character" as const, palette_size: 24, seamless: false, symmetric: false, dither: false, seed: null, params: overrides };
    const { job_id } = await startAnimation(spec ? {
      ...base, motion: "custom", poses: spec.poses, motion_text: spec.text || undefined, fps: spec.fps, anchor: spec.anchor,
      allow_objects: spec.allowObjects, extra_avoid: spec.extraAvoid || undefined,
    } : { ...base, frames: 8 });
    setJob({ progress: 0, total: 1, elapsed: 0 });
    try {
      for (;;) {
        await sleep(1500);
        const j = await getJob<AnimationResponse>(job_id);
        setJob({ progress: j.progress, total: j.total, elapsed: j.elapsed });
        if (j.status === "error") throw new Error(j.error ?? "A animação falhou.");
        if (j.status === "done" && j.result) {
          setAnim({ frames: await Promise.all(j.result.frames.map((f) => pngToImageData(f.image_b64))), fps: j.result.fps });
          return;
        }
      }
    } finally { setJob(null); }
  }

  // o editor de poses entrega o movimento aqui: a geração roda no painel (barra de progresso e resultado)
  useEffect(() => {
    if (!custom) return;
    onCustomStarted();
    setTab("animate");
    void run(() => runAnimation(custom));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [custom]);

  const tabCls = (t: Tab) =>
    `flex items-center gap-1 px-3 py-1.5 text-sm rounded-md ${tab === t ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`;
  const btn = "flex items-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-3 py-1.5 text-sm text-white";

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-[min(720px,calc(100%-2rem))] z-10 rounded-xl border border-zinc-800 bg-zinc-900/95 backdrop-blur shadow-2xl">
      <div className="flex gap-1 p-2 border-b border-zinc-800">
        <button className={tabCls("animate")} onClick={() => setTab("animate")}><Film size={14} /> Animate</button>
        <button className={tabCls("edit")} onClick={() => setTab("edit")}><Pencil size={14} /> Edit</button>
        <button className={tabCls("map")} onClick={() => setTab("map")}><MapIcon size={14} /> Map Blueprint</button>
      </div>
      <div className="p-3 max-h-72 overflow-y-auto text-sm">
        {error && <p className="mb-2 text-red-300">{error}</p>}
        {tab === "animate" && (
          <div className="space-y-2">
            <button className={btn} disabled={busy} onClick={() => run(async () => {
              need();
              const r = await generate8({ prompt, size, kind: "character", palette_size: 16, seamless: false, symmetric: false, dither: false, seed: sprite?.seed ?? null });
              const d = normalize8(r);
              if (!d.length) throw new Error("O backend não retornou direções válidas.");
              setDirs(d);
            })}>
              {busy ? <Loader2 size={14} className="animate-spin" /> : <Film size={14} />} Gerar 8 direções
            </button>
            <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800 pt-2">
              <button className={btn} disabled={busy} onClick={() => run(() => runAnimation())}>
                {busy && job ? <Loader2 size={14} className="animate-spin" /> : <Footprints size={14} />} Gerar corrida (8 frames)
              </button>
              <button className={btn} disabled={busy} onClick={openPoseEditor}>
                <PersonStanding size={14} /> Animação personalizada (poses)
              </button>
              {anim && (
                <button className="flex items-center gap-2 rounded-lg border border-emerald-700 px-3 py-1.5 text-sm text-emerald-300 hover:bg-emerald-950"
                  onClick={openEditor}><PencilRuler size={14} /> Correção manual</button>
              )}
              {job && (
                <div className="flex-1 min-w-40" role="progressbar" aria-valuemin={0} aria-valuemax={job.total} aria-valuenow={job.progress}>
                  <div className="h-2 overflow-hidden rounded bg-zinc-800">
                    <div className="h-full bg-emerald-500 transition-all" style={{ width: `${Math.round((job.progress / Math.max(1, job.total)) * 100)}%` }} />
                  </div>
                  <p className="mt-1 text-xs text-zinc-500">{job.progress}/{job.total} imagens · {Math.round(job.elapsed)} s</p>
                </div>
              )}
            </div>
            {anim && (
              <div className="flex items-center gap-3">
                <AnimPreview anim={anim} />
                <p className="text-xs text-zinc-500">{anim.frames.length} frames · {anim.fps} fps · {anim.frames[0].width}px.
                  Abra a correção manual para ver frame a frame, desenhar e exportar o GIF.</p>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              {dirs.map(([n, s]) => (
                <figure key={n} className="text-center text-xs text-zinc-500">
                  <img src={toDataUrl(s)} alt={n} className="pixelated w-16 h-16 bg-zinc-800 rounded" />{n}
                </figure>
              ))}
            </div>
          </div>
        )}
        {tab === "edit" && (
          <p className="text-zinc-500">Edição (inpainting) ainda não disponível nesta versão. Gere um sprite e ajuste o prompt para variações.</p>
        )}
        {tab === "map" && (
          <div className="space-y-2">
            <button className={btn} disabled={busy} onClick={() => run(async () => {
              need(); setMap(await generateMap(prompt));
            })}>
              {busy ? <Loader2 size={14} className="animate-spin" /> : <MapIcon size={14} />} Gerar mapa completo
            </button>
            {map && (
              <>
                <div className="flex gap-1">
                  {map.master_palette.map((c) => <span key={c} title={c} className="w-5 h-5 rounded border border-zinc-700" style={{ background: c }} />)}
                </div>
                <pre className="whitespace-pre-wrap text-xs text-zinc-300 bg-zinc-950 rounded p-2">{map.manual_markdown}</pre>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
