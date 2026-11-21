import { useEffect, useState } from "react";
import { Settings } from "lucide-react";
import Sidebar, { type Mode } from "./components/Sidebar";
import Canvas from "./components/Canvas";
import FloatingPanel from "./components/FloatingPanel";
import AnimResult from "./components/AnimResult";
import FrameEditor from "./components/FrameEditor";
import PoseEditor from "./components/PoseEditor";
import { useAnimation } from "./useAnimation";
import { loadPalette } from "./components/PaletteChooser";
import SettingsPopover, { type ViewSettings } from "./components/SettingsPopover";
import { generate, getParams, type GenOverrides, type ParamsInfo, type SpriteResponse } from "./api";

const OVERRIDES_KEY = "openpixel.overrides";

function loadOverrides(): GenOverrides {
  try {
    const v = JSON.parse(localStorage.getItem(OVERRIDES_KEY) ?? "{}");
    return v && typeof v === "object" ? v : {};
  } catch { return {}; }
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState(32);
  const [sprite, setSprite] = useState<SpriteResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [overrides, setOverrides] = useState<GenOverrides>(loadOverrides);
  const [info, setInfo] = useState<ParamsInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("create");
  const [palette, setPalette] = useState<string[] | null>(loadPalette);
  const [editing, setEditing] = useState(false);
  const { anim, setAnim, busy: animBusy, job, error: animError, run: runAnimation } = useAnimation(prompt, size, overrides, palette);
  const [view, setView] = useState<ViewSettings>({ crt: false, checkerboard: true, pixelPerfect: true });

  useEffect(() => {
    getParams().then(setInfo).catch((e: Error) => setInfoError(e.message));
  }, []);
  useEffect(() => {
    try { localStorage.setItem(OVERRIDES_KEY, JSON.stringify(overrides)); } catch { /* armazenamento indisponível */ }
  }, [overrides]);

  async function onGenerate() {
    if (!prompt.trim()) { setError("Descreva o que deseja gerar."); return; }
    setBusy(true); setError(null);
    try {
      setSprite(await generate({
        prompt, size, kind: "asset", palette_size: 16,
        seamless: false, symmetric: false, dither: false, seed: null, params: overrides,
        custom_palette: palette ?? undefined,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erro desconhecido.");
    } finally { setBusy(false); }
  }

  return (
    <div className="flex h-full bg-zinc-950">
      <Sidebar mode={mode} setMode={setMode} palette={palette} setPalette={setPalette} prompt={prompt} setPrompt={setPrompt} size={size} setSize={setSize} busy={busy} onGenerate={onGenerate} />
      <main className="relative flex-1 min-w-0">
        <div className="absolute top-3 right-3 z-20">
          <button aria-label="Settings" onClick={() => setShowSettings((v) => !v)}
            className="p-2 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800">
            <Settings size={18} />
          </button>
          {showSettings && <SettingsPopover value={view} onChange={setView} info={info} infoError={infoError}
            overrides={overrides} setOverrides={setOverrides} />}
        </div>
        {mode !== "animate" && (
          <>
            <Canvas sprite={sprite} busy={busy} error={error} view={view} />
            <FloatingPanel prompt={prompt} size={size} sprite={sprite} overrides={overrides} />
          </>
        )}
        {/* sempre montado: trocar de aba não perde o histórico de desfazer do editor de poses */}
        <div className={mode === "animate" ? "h-full" : "hidden"}>
          <PoseEditor prompt={prompt} active={mode === "animate"} busy={animBusy} onGenerate={runAnimation}
            result={<AnimResult anim={anim} job={job} error={animError} openEditor={() => setEditing(true)} />} />
        </div>
      </main>
      {editing && anim && (
        <FrameEditor frames={anim.frames} fps={anim.fps} onClose={() => setEditing(false)}
          onApply={(frames, fps) => { setAnim({ frames, fps }); setEditing(false); }} />
      )}
    </div>
  );
}
