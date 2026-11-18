import { useState } from "react";
import { Settings } from "lucide-react";
import Sidebar from "./components/Sidebar";
import Canvas from "./components/Canvas";
import FloatingPanel from "./components/FloatingPanel";
import SettingsPopover, { type ViewSettings } from "./components/SettingsPopover";
import { generate, type SpriteResponse } from "./api";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState(32);
  const [sprite, setSprite] = useState<SpriteResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [view, setView] = useState<ViewSettings>({ crt: false, checkerboard: true, pixelPerfect: true });

  async function onGenerate() {
    if (!prompt.trim()) { setError("Descreva o que deseja gerar."); return; }
    setBusy(true); setError(null);
    try {
      setSprite(await generate({
        prompt, size, kind: "asset", palette_size: 16,
        seamless: false, symmetric: false, dither: false, seed: null,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erro desconhecido.");
    } finally { setBusy(false); }
  }

  return (
    <div className="flex h-full bg-zinc-950">
      <Sidebar prompt={prompt} setPrompt={setPrompt} size={size} setSize={setSize} busy={busy} onGenerate={onGenerate} />
      <main className="relative flex-1 min-w-0">
        <div className="absolute top-3 right-3 z-20">
          <button aria-label="Settings" onClick={() => setShowSettings((v) => !v)}
            className="p-2 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800">
            <Settings size={18} />
          </button>
          {showSettings && <SettingsPopover value={view} onChange={setView} />}
        </div>
        <Canvas sprite={sprite} busy={busy} error={error} view={view} />
        <FloatingPanel prompt={prompt} size={size} sprite={sprite} />
      </main>
    </div>
  );
}
