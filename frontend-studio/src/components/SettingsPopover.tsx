import { useState } from "react";
import { RotateCcw } from "lucide-react";
import type { GenOverrides, ParamsInfo } from "../api";

export interface ViewSettings { crt: boolean; checkerboard: boolean; pixelPerfect: boolean }

const ITEMS: [keyof ViewSettings, string][] = [
  ["crt", "CRT Mode"],
  ["checkerboard", "Checkerboard"],
  ["pixelPerfect", "Pixel Perfect"],
];

interface Props {
  value: ViewSettings; onChange: (v: ViewSettings) => void;
  info: ParamsInfo | null; infoError: string | null;
  overrides: GenOverrides; setOverrides: (o: GenOverrides) => void;
}

const fmt = (n: number) => (Math.abs(n) >= 10 || Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, "").replace(/\.$/, ""));

function GenerationTab({ info, infoError, overrides, setOverrides }: Omit<Props, "value" | "onChange">) {
  if (infoError) return <p className="text-xs text-red-300">{infoError}</p>;
  if (!info) return <p className="text-xs text-zinc-500">Carregando fatores...</p>;
  const groups = [...new Set(info.specs.map((s) => s.group))];
  const edited = Object.keys(overrides).length;
  const set = (key: string, v: number) => {
    const next = { ...overrides };
    if (v === info.defaults[key]) delete next[key]; else next[key] = v;
    setOverrides(next);
  };
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-zinc-500">
        <span>Provider: {info.provider}</span>
        <button disabled={!edited} onClick={() => setOverrides({})}
          className="flex items-center gap-1 text-zinc-300 disabled:text-zinc-600 hover:text-white">
          <RotateCcw size={12} /> Restaurar padrões{edited ? ` (${edited})` : ""}
        </button>
      </div>
      {info.provider !== "comfyui_sdxl_pixel" && (
        <p className="rounded bg-amber-950/60 border border-amber-900 p-2 text-xs text-amber-200">
          O provider atual não é o ComfyUI: LoRAs, CFG, passos, pose e referência só têm efeito com DEVICE_MODE=comfy.
        </p>
      )}
      {groups.map((g) => (
        <section key={g} className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">{g}</h4>
          {info.specs.filter((s) => s.group === g).map((s) => {
            const v = overrides[s.key] ?? info.defaults[s.key];
            const changed = s.key in overrides;
            return (
              <div key={s.key} title={s.help}>
                <div className="flex items-center justify-between text-xs">
                  <span className={changed ? "text-emerald-300" : "text-zinc-300"}>{s.label}</span>
                  <input type="number" value={v} min={s.min} max={s.max} step={s.step} aria-label={s.label}
                    onChange={(e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) set(s.key, Math.min(s.max, Math.max(s.min, n))); }}
                    className="w-16 rounded bg-zinc-950 border border-zinc-800 px-1 py-0.5 text-right text-zinc-100" />
                </div>
                <input type="range" value={v} min={s.min} max={s.max} step={s.step} aria-label={`${s.label} (controle)`}
                  onChange={(e) => set(s.key, Number(e.target.value))} className="w-full accent-emerald-500" />
                <div className="flex justify-between text-[10px] text-zinc-600">
                  <span>{fmt(s.min)}</span><span>padrão {fmt(info.defaults[s.key])}</span><span>{fmt(s.max)}</span>
                </div>
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}

export default function SettingsPopover(props: Props) {
  const [tab, setTab] = useState<"view" | "gen">("gen");
  const tabCls = (t: string) =>
    `flex-1 py-1.5 text-sm border-b-2 ${tab === t ? "border-emerald-500 text-zinc-100" : "border-transparent text-zinc-500 hover:text-zinc-300"}`;
  return (
    <div className="absolute right-0 mt-2 w-80 max-h-[80vh] overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-900 p-3 shadow-xl">
      <div className="flex mb-3 border-b border-zinc-800">
        <button className={tabCls("gen")} onClick={() => setTab("gen")}>Geração</button>
        <button className={tabCls("view")} onClick={() => setTab("view")}>Visual</button>
      </div>
      {tab === "view" ? (
        <div className="space-y-2">
          {ITEMS.map(([k, label]) => (
            <label key={k} className="flex items-center justify-between text-sm cursor-pointer">
              {label}
              <input type="checkbox" checked={props.value[k]} className="accent-emerald-500"
                onChange={(e) => props.onChange({ ...props.value, [k]: e.target.checked })} />
            </label>
          ))}
        </div>
      ) : <GenerationTab info={props.info} infoError={props.infoError} overrides={props.overrides} setOverrides={props.setOverrides} />}
    </div>
  );
}
