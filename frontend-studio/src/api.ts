export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type AssetKind = "character" | "asset" | "tileset" | "map";

export interface GenerateRequest {
  prompt: string;
  size: number;
  kind: AssetKind;
  palette_size: number;
  seamless: boolean;
  symmetric: boolean;
  dither: boolean;
  seed?: number | null;
}

export interface SpriteResponse {
  image_b64: string;
  width: number;
  height: number;
  palette: string[];
  seed: number;
  provider: string;
}

export interface MapAsset { name: string; kind: "tile" | "prop"; prompt: string; size: number }

export interface CompleteMapResponse {
  master_palette: string[];
  asset_manifest: MapAsset[];
  tilemap_layers: Record<string, number[][]>;
  composite_map_path: string;
  manual_path: string;
  manual_markdown: string;
  assets?: Record<string, SpriteResponse>;
}

export interface VramInfo { mode: string; used_mb: number; total_mb: number }

export interface DatasetItem {
  id: number | string;
  source: string;
  path: string;
  caption_path: string | null;
  kind: string;
  width: number;
  height: number;
  approved: boolean | number;
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(`Não foi possível conectar ao backend em ${API_URL}. Verifique se ele está rodando.`);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? "");
    } catch { /* ignore */ }
    throw new ApiError(`O servidor respondeu com erro ${res.status}${detail ? `: ${detail}` : "."}`);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError("Resposta inválida do servidor.");
  }
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

export const generate = (b: GenerateRequest) => post<SpriteResponse>("/generate", b);
export const generate8 = (b: GenerateRequest) => post<unknown>("/generate8", b);
export const generateMap = (prompt: string) => post<CompleteMapResponse>("/map", { prompt });
export const getVram = () => request<VramInfo>("/vram");
export const getDataset = () => request<DatasetItem[]>("/dataset");

export const toDataUrl = (s: SpriteResponse) => `data:image/png;base64,${s.image_b64}`;
