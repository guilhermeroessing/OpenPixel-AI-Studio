import { Plus, X } from "lucide-react";

export const MIN_COLORS = 2;
export const MAX_COLORS = 8;
export const PALETTE_KEY = "openpixel.palette";

/** Paletas prontas, só para começar: o usuário troca cada cor. */
const PRESETS: { label: string; colors: string[] }[] = [
  { label: "2 cores", colors: ["#1f2a44", "#f4e3c1"] },
  { label: "Game Boy (4)", colors: ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"] },
  { label: "Terra (6)", colors: ["#f1c27d", "#8b4a2b", "#3f7d4e", "#2d3a63", "#5a3a22", "#f2efe6"] },
];

export function loadPalette(): string[] | null {
  try {
    const v = JSON.parse(localStorage.getItem(PALETTE_KEY) ?? "null");
    if (Array.isArray(v) && v.length >= MIN_COLORS && v.length <= MAX_COLORS && v.every((c) => /^#[0-9a-f]{6}$/i.test(c))) return v;
  } catch { /* armazenamento indisponível */ }
  return null;
}

/**
 * Paleta personalizada (2 a 8 cores): desligada = a IA escolhe as cores; ligada = o desenho é composto só com estas.
 * `palette === null` significa desligada.
 */
export default function PaletteChooser({ palette, setPalette }: { palette: string[] | null; setPalette: (p: string[] | null) => void }) {
  const on = palette !== null;
  const set = (p: string[] | null) => {
    setPalette(p);
    try { if (p) localStorage.setItem(PALETTE_KEY, JSON.stringify(p)); else localStorage.removeItem(PALETTE_KEY); } catch { /* sem armazenamento */ }
  };
  const colors = palette ?? [];
  return (
    <div className="space-y-2 text-xs text-zinc-400">
      <label className="flex items-center justify-between">
        <span>Paleta personalizada ({MIN_COLORS} a {MAX_COLORS} cores)</span>
        <input type="checkbox" checked={on} className="accent-emerald-500" aria-label="Usar paleta personalizada"
          onChange={(e) => set(e.target.checked ? PRESETS[2].colors : null)} />
      </label>
      {on && (
        <>
          <div className="flex flex-wrap items-center gap-1.5" aria-label="Cores da paleta">
            {colors.map((c, i) => (
              <span key={i} className="group relative">
                <input type="color" value={c} aria-label={`Cor ${i + 1}`}
                  onChange={(e) => set(colors.map((x, k) => (k === i ? e.target.value : x)))}
                  className="h-8 w-8 cursor-pointer rounded border border-zinc-700 bg-transparent p-0" />
                {colors.length > MIN_COLORS && (
                  <button aria-label={`Remover cor ${i + 1}`} onClick={() => set(colors.filter((_, k) => k !== i))}
                    className="absolute -right-1 -top-1 hidden rounded-full bg-zinc-900 text-zinc-300 hover:text-white group-hover:block focus:block">
                    <X size={12} />
                  </button>
                )}
              </span>
            ))}
            {colors.length < MAX_COLORS && (
              <button aria-label="Adicionar cor" onClick={() => set([...colors, "#808080"])}
                className="flex h-8 w-8 items-center justify-center rounded border border-dashed border-zinc-700 text-zinc-400 hover:bg-zinc-800">
                <Plus size={14} />
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {PRESETS.map((p) => (
              <button key={p.label} onClick={() => set(p.colors)}
                className="rounded border border-zinc-800 bg-zinc-900 px-2 py-0.5 text-[11px] text-zinc-300 hover:bg-zinc-800">{p.label}</button>
            ))}
          </div>
          <p className="text-zinc-600">{colors.length} cores. O desenho e as animações usam só estas cores. Com 2 ou 3 cores de pouco contraste de matiz (ex.: escuro e claro) a divisão é por luminosidade.</p>
        </>
      )}
    </div>
  );
}
