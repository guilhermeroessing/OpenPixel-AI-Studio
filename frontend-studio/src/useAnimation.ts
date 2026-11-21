import { useState } from "react";
import { getJob, startAnimation, type AnimationResponse, type GenOverrides } from "./api";
import { pngToImageData } from "./animation";
import type { CustomMotionSpec } from "./components/PoseEditor";

export interface AnimState { frames: ImageData[]; fps: number }
export interface JobProgress { progress: number; total: number; elapsed: number }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Gera uma animação personalizada em segundo plano (POST /animate + GET /jobs) e guarda o resultado. */
export function useAnimation(prompt: string, size: number, overrides: GenOverrides, palette: string[] | null) {
  const [anim, setAnim] = useState<AnimState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<JobProgress | null>(null);

  async function run(spec: CustomMotionSpec) {
    if (!prompt.trim()) { setError("Escreva o personagem no prompt da barra lateral primeiro."); return; }
    setBusy(true); setError(null);
    try {
      const { job_id } = await startAnimation({
        prompt, size, kind: "character", palette_size: 24, seamless: false, symmetric: false, dither: false,
        seed: null, params: overrides, custom_palette: palette ?? undefined, motion: "custom", poses: spec.poses, motion_text: spec.text || undefined,
        fps: spec.fps, anchor: spec.anchor, allow_objects: spec.allowObjects, extra_avoid: spec.extraAvoid || undefined,
      });
      setJob({ progress: 0, total: 1, elapsed: 0 });
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
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erro desconhecido.");
    } finally { setJob(null); setBusy(false); }
  }

  return { anim, setAnim, busy, job, error, run };
}
