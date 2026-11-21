import { useCallback, useEffect, useState } from "react";
import { Database, Film, Loader2, Sparkles, Wand2, Cpu } from "lucide-react";
import { getDataset, getVram, type DatasetItem, type VramInfo } from "../api";

const SIZES = [16, 32, 64, 128];

export type Mode = "create" | "animate" | "dataset";

interface Props {
  mode: Mode; setMode: (m: Mode) => void;
  prompt: string; setPrompt: (s: string) => void;
  size: number; setSize: (n: number) => void;
  busy: boolean; onGenerate: () => void;
}

function VramMonitor() {
  const [vram, setVram] = useState<VramInfo | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    let alive = true;
    const tick = () => getVram()
      .then((v) => { if (alive) { setVram(v); setErr(false); } })
      .catch(() => { if (alive) setErr(true); });
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  const pct = vram && vram.total_mb > 0 ? Math.min(100, (vram.used_mb / vram.total_mb) * 100) : 0;
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3">
      <div className="flex items-center justify-between text-xs text-zinc-400 mb-2">
        <span className="flex items-center gap-1"><Cpu size={14} /> VRAM</span>
        <span>{vram ? vram.mode : "-"}</span>
      </div>
      <div className="h-2 rounded bg-zinc-800 overflow-hidden">
        <div className="h-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-1 text-xs text-zinc-500">
        {err ? "Backend offline: VRAM indisponível." : vram ? `${vram.used_mb} / ${vram.total_mb} MB` : "Carregando..."}
      </p>
    </div>
  );
}

function DatasetTab() {
  const [items, setItems] = useState<DatasetItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    setError(null); setItems(null);
    getDataset().then(setItems).catch((e: Error) => setError(e.message));
  }, []);
  useEffect(load, [load]);
  return (
    <div className="space-y-2">
      {error && (
        <div className="text-xs text-red-300">
          {error} <button className="underline" onClick={load}>Tentar novamente</button>
        </div>
      )}
      {!error && items === null && <Loader2 className="animate-spin text-zinc-500" size={18} />}
      {items?.length === 0 && <p className="text-xs text-zinc-500">Nenhum item no dataset.</p>}
      <ul className="space-y-1">
        {items?.map((it) => (
          <li key={it.id} className="rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs">
            <div className="flex justify-between">
              <span className="truncate text-zinc-300" title={it.path}>{it.path.split(/[\\/]/).pop()}</span>
              <span className={it.approved ? "text-emerald-400" : "text-zinc-500"}>{it.approved ? "ok" : "pend."}</span>
            </div>
            <div className="text-zinc-500">{it.source} · {it.kind} · {it.width}x{it.height}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Sidebar({ mode: tab, setMode: setTab, prompt, setPrompt, size, setSize, busy, onGenerate }: Props) {
  const tabCls = (t: string) =>
    `flex-1 flex items-center justify-center gap-1 py-2 text-sm border-b-2 ${tab === t ? "border-emerald-500 text-zinc-100" : "border-transparent text-zinc-500 hover:text-zinc-300"}`;
  return (
    <aside className="w-72 shrink-0 h-full flex flex-col border-r border-zinc-800 bg-zinc-950">
      <div className="px-4 py-3 font-semibold flex items-center gap-2"><Sparkles size={18} className="text-emerald-400" /> OpenPixel Studio</div>
      <div className="flex border-b border-zinc-800">
        <button className={tabCls("create")} onClick={() => setTab("create")}><Wand2 size={14} /> Criar</button>
        <button className={tabCls("animate")} onClick={() => setTab("animate")}><Film size={14} /> Animar</button>
        <button className={tabCls("dataset")} onClick={() => setTab("dataset")}><Database size={14} /> Dataset</button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {tab !== "dataset" ? (
          <>
            <label className="block text-xs text-zinc-400">Prompt
              <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={5}
                placeholder={tab === "animate" ? "ex.: mulher esguia, cabelo ciano curto, regata branca" : "ex.: espada de fogo, estilo 16-bit"}
                className="mt-1 w-full resize-none rounded-lg bg-zinc-900 border border-zinc-800 p-2 text-sm text-zinc-100 focus:outline-none focus:border-emerald-500" />
            </label>
            <div>
              <div className="text-xs text-zinc-400 mb-1">Tamanho</div>
              <div className="flex gap-2">
                {SIZES.map((s) => (
                  <button key={s} onClick={() => setSize(s)}
                    className={`flex-1 rounded-full py-1 text-sm border ${size === s ? "bg-emerald-600 border-emerald-500 text-white" : "border-zinc-800 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"}`}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
            {tab === "create" ? (
              <button onClick={onGenerate} disabled={busy}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 py-2 text-sm font-medium text-white">
                {busy ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />} Gerar
              </button>
            ) : (
              <p className="text-xs text-zinc-500">Aqui você descreve só o personagem. O movimento é montado ao lado, pose a pose; o tamanho 128 dá o melhor resultado.</p>
            )}
          </>
        ) : <DatasetTab />}
      </div>
      <div className="p-4 border-t border-zinc-800"><VramMonitor /></div>
    </aside>
  );
}
