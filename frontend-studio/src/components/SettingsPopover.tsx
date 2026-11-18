export interface ViewSettings { crt: boolean; checkerboard: boolean; pixelPerfect: boolean }

const ITEMS: [keyof ViewSettings, string][] = [
  ["crt", "CRT Mode"],
  ["checkerboard", "Checkerboard"],
  ["pixelPerfect", "Pixel Perfect"],
];

export default function SettingsPopover({ value, onChange }: { value: ViewSettings; onChange: (v: ViewSettings) => void }) {
  return (
    <div className="absolute right-0 mt-2 w-56 rounded-lg border border-zinc-800 bg-zinc-900 p-3 shadow-xl space-y-2">
      {ITEMS.map(([k, label]) => (
        <label key={k} className="flex items-center justify-between text-sm cursor-pointer">
          {label}
          <input type="checkbox" checked={value[k]} className="accent-emerald-500"
            onChange={(e) => onChange({ ...value, [k]: e.target.checked })} />
        </label>
      ))}
    </div>
  );
}
