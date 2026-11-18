import { useState } from "react";
import { Film, Loader2, Map as MapIcon, Pencil } from "lucide-react";
import { generate8, generateMap, toDataUrl, type CompleteMapResponse, type SpriteResponse } from "../api";

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

export default function FloatingPanel({ prompt, size, sprite }: { prompt: string; size: number; sprite: SpriteResponse | null }) {
  const [tab, setTab] = useState<Tab>("animate");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirs, setDirs] = useState<[string, SpriteResponse][]>([]);
  const [map, setMap] = useState<CompleteMapResponse | null>(null);

  async function run(fn: () => Promise<void>) {
    setBusy(true); setError(null);
    try { await fn(); } catch (e) { setError(e instanceof Error ? e.message : "Erro desconhecido."); }
    finally { setBusy(false); }
  }
  const need = () => { if (!prompt.trim()) throw new Error("Escreva um prompt na barra lateral primeiro."); };

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
      <div className="p-3 max-h-56 overflow-y-auto text-sm">
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
