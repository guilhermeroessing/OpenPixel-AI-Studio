import { useEffect, useRef } from "react";
import { PencilRuler } from "lucide-react";
import { imageDataToCanvas } from "../animation";
import type { AnimState, JobProgress } from "../useAnimation";

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
    className="pixelated checker h-32 w-32 shrink-0 rounded border border-zinc-800" />;
}

/** Andamento, erro e resultado da animação (prévia + atalho para a correção manual). */
export default function AnimResult({ anim, job, error, openEditor }: {
  anim: AnimState | null; job: JobProgress | null; error: string | null; openEditor: () => void;
}) {
  return (
    <div className="space-y-2" aria-live="polite">
      {error && <p className="text-red-300">{error}</p>}
      {job && (
        <div role="progressbar" aria-valuemin={0} aria-valuemax={job.total} aria-valuenow={job.progress}>
          <div className="h-2 overflow-hidden rounded bg-zinc-800">
            <div className="h-full bg-emerald-500 transition-all" style={{ width: `${Math.round((job.progress / Math.max(1, job.total)) * 100)}%` }} />
          </div>
          <p className="mt-1 text-zinc-500">{job.progress}/{job.total} imagens · {Math.round(job.elapsed)} s</p>
        </div>
      )}
      {anim && (
        <div className="flex items-center gap-3">
          <AnimPreview anim={anim} />
          <div className="space-y-2">
            <p className="text-zinc-500">{anim.frames.length} frames · {anim.fps} fps · {anim.frames[0].width}px</p>
            <button className="flex items-center gap-2 rounded-lg border border-emerald-700 px-3 py-1.5 text-sm text-emerald-300 hover:bg-emerald-950"
              onClick={openEditor}><PencilRuler size={14} /> Correção manual</button>
          </div>
        </div>
      )}
    </div>
  );
}
