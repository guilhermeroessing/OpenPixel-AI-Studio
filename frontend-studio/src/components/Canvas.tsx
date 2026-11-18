import { AlertTriangle, Loader2 } from "lucide-react";
import { toDataUrl, type SpriteResponse } from "../api";
import type { ViewSettings } from "./SettingsPopover";

export default function Canvas({ sprite, busy, error, view }: {
  sprite: SpriteResponse | null; busy: boolean; error: string | null; view: ViewSettings;
}) {
  // Pixel Perfect: escala inteira que cabe em ~512px.
  const scale = sprite ? Math.max(1, Math.floor(512 / Math.max(sprite.width, sprite.height))) : 1;
  return (
    <div className={`h-full flex items-center justify-center p-6 pb-40 ${view.checkerboard ? "checker" : "bg-zinc-900"}`}>
      {error && (
        <div className="absolute top-3 left-3 right-16 flex items-center gap-2 rounded-lg border border-red-900 bg-red-950/80 px-3 py-2 text-sm text-red-200">
          <AlertTriangle size={16} /> {error}
        </div>
      )}
      {busy && <Loader2 className="animate-spin text-zinc-400" size={32} />}
      {!busy && sprite && (
        <div className={view.crt ? "crt" : ""}>
          <img src={toDataUrl(sprite)} alt="Sprite gerado" className="pixelated block"
            style={view.pixelPerfect
              ? { width: sprite.width * scale, height: sprite.height * scale }
              : { width: "min(70vh, 100%)", height: "auto" }} />
        </div>
      )}
      {!busy && !sprite && <p className="text-zinc-500 text-sm bg-zinc-950/70 rounded px-3 py-2">Escreva um prompt e clique em Gerar.</p>}
    </div>
  );
}
