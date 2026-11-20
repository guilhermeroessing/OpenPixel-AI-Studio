"""ComfyProvider: gera pela API local do ComfyUI (http://127.0.0.1:8188) e pixeliza no backend.

Mesma interface do SmartMockProvider: generate(prompt, size, seed) -> (RGB uint8 HxWx3, seed).
Usa so a biblioteca padrao (urllib) + PIL/numpy. O ComfyUI mantem o modelo carregado entre
geracoes e cuida da memoria.

Configuracao (variaveis de ambiente ou .env):
  COMFY_URL         endereco do ComfyUI              (padrao http://127.0.0.1:8188)
  COMFY_CHECKPOINT  checkpoint SDXL                  (padrao sd_xl_base_1.0.safetensors)
  COMFY_LORAS       pilha de LoRAs "arquivo:forca,arquivo:forca" (padrao: pixel-art-xl 0.7 + PixelArtRedmond-Lite64 0.7)
                    use "none" para gerar so com o checkpoint
  COMFY_TRIGGER     palavras-gatilho das LoRAs       (padrao "Pixel Art, PixArFK, pixel art")
  COMFY_AVOID       termos extras para o prompt negativo (ex.: "(weapon:1.6), (shield:1.8)")
  COMFY_STEPS / COMFY_CFG / COMFY_SIZE   passos, cfg e lado da imagem (25 / 6.5 / 1024)
  COMFY_POSE        pose padrao (ControlNet OpenPose): nome de preset (idle, hands_on_hips, arms_up, walk) ou
                    caminho de uma imagem de esqueleto; vazio = sem controle de pose
  COMFY_POSE_STRENGTH / COMFY_CONTROLNET   forca (1.0) e arquivo do ControlNet (xinsir_openpose_sdxl.safetensors)
  COMFY_REF_WEIGHT  peso do IP-Adapter quando generate(..., reference=imagem) e usado (0.8)
  PIXEL_GRID        grade logica de uma pixelizacao opcional dentro do provider (padrao 0 = desligada:
                    o service.py ja pixeliza com openpixel_core.pixelate (PIXELIZER=box, padrao);
                    ligue so se quiser pixelizar antes, ex. 128)
  PIXEL_COLORS      cores da paleta adaptativa dessa pixelizacao opcional (padrao 24)
"""
from __future__ import annotations

import copy
import io
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
from PIL import Image

from .pixelate import pixelate
from .poses import PRESETS, preset_image

NEGATIVE = "blurry, anti-aliasing, smooth gradient, photo, realistic, 3d render, jpeg artifacts, text, watermark"
DEFAULT_LORAS = "pixel-art-xl-v1.1.safetensors:0.7,PixelArtRedmond-Lite64.safetensors:0.7"
DEFAULT_TRIGGER = "Pixel Art, PixArFK, pixel art"


def parse_loras(spec: str) -> list[tuple[str, float]]:
    """'a.safetensors:0.7,b.safetensors' -> [('a.safetensors', 0.7), ('b.safetensors', 1.0)]; 'none' -> []."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "none":
        return []
    out = []
    for item in spec.split(","):
        name, sep, strength = item.strip().rpartition(":")
        if not sep:  # sem ":forca"
            name, strength = strength, "1.0"
        out.append((name.strip(), float(strength)))
    return out


class ComfyProvider:
    name = "comfyui_sdxl_pixel"

    def __init__(self, url: str | None = None, checkpoint: str | None = None, steps: int | None = None,
                 cfg: float | None = None, native_size: int | None = None, loras: str | None = None,
                 trigger: str | None = None, avoid: str | None = None, grid: int | None = None,
                 colors: int | None = None, pose: str | None = None, pose_strength: float | None = None,
                 controlnet: str | None = None, timeout: int = 600):
        env = os.environ.get
        self.url = (url or env("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        self.checkpoint = checkpoint or env("COMFY_CHECKPOINT", "sd_xl_base_1.0.safetensors")
        self.steps = steps or int(env("COMFY_STEPS", "25"))
        self.cfg = cfg or float(env("COMFY_CFG", "6.5"))
        self.native_size = native_size or int(env("COMFY_SIZE", "1024"))
        self.loras = parse_loras(loras if loras is not None else env("COMFY_LORAS", DEFAULT_LORAS))
        self.trigger = trigger if trigger is not None else env("COMFY_TRIGGER", DEFAULT_TRIGGER)
        self.avoid = avoid if avoid is not None else env("COMFY_AVOID", "")
        self.grid = grid if grid is not None else int(env("PIXEL_GRID", "0"))
        self.colors = colors or int(env("PIXEL_COLORS", "24"))
        self.pose = pose if pose is not None else env("COMFY_POSE", "")
        self.pose_strength = pose_strength if pose_strength is not None else float(env("COMFY_POSE_STRENGTH", "1.0"))
        self.ref_weight = float(env("COMFY_REF_WEIGHT", "0.8"))
        self.controlnet = controlnet or env("COMFY_CONTROLNET", "xinsir_openpose_sdxl.safetensors")
        self.timeout = timeout
        self._uploaded: dict[str, str] = {}

    # ---- HTTP ---------------------------------------------------------------
    def _http(self, path: str, data: bytes | None = None, raw: bool = False, timeout: int = 60):
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI HTTP {exc.code} em {path}: {exc.read().decode(errors='replace')[:800]}") from exc
        except OSError as exc:
            raise RuntimeError(f"Nao consegui falar com o ComfyUI em {self.url} ({exc}). Ele esta aberto?") from exc
        return body if raw else json.loads(body)

    def with_overrides(self, cfg: float | None = None, steps: int | None = None, pose_strength: float | None = None,
                       ref_weight: float | None = None, lora_strengths: dict[str, float] | None = None,
                       avoid: str | None = None) -> "ComfyProvider":
        """Copia do provider com forcas trocadas (por requisicao, sem mexer na instancia global).

        lora_strengths: {trecho do nome da LoRA (sem diferenciar maiusculas): forca}.
        """
        c = copy.copy(self)
        if cfg is not None:
            c.cfg = cfg
        if steps is not None:
            c.steps = steps
        if pose_strength is not None:
            c.pose_strength = pose_strength
        if ref_weight is not None:
            c.ref_weight = ref_weight
        if avoid is not None:
            c.avoid = avoid
        if lora_strengths:
            def pick(name: str, old: float) -> float:
                for key, val in lora_strengths.items():
                    if key.lower() in name.lower():
                        return val
                return old
            c.loras = [(n, pick(n, s)) for n, s in self.loras]
        return c

    def is_up(self) -> bool:
        try:
            self._http("/system_stats", timeout=3)
            return True
        except RuntimeError:
            return False

    def _upload_pose(self, pose: str) -> str:
        """Envia a imagem de pose (preset ou arquivo) para o input do ComfyUI; devolve o nome salvo."""
        if pose in self._uploaded:
            return self._uploaded[pose]
        if pose in PRESETS:
            img, fname = Image.fromarray(preset_image(pose, self.native_size)), f"openpixel_pose_{pose}_{self.native_size}.png"
        else:
            img, fname = Image.open(pose).convert("RGB"), f"openpixel_pose_{uuid.uuid5(uuid.NAMESPACE_URL, pose).hex[:10]}.png"
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        crlf = "\r\n"
        head = (f"--{boundary}{crlf}Content-Disposition: form-data; name=\"image\"; filename=\"{fname}\"{crlf}"
                f"Content-Type: image/png{crlf}{crlf}")
        tail = (f"{crlf}--{boundary}{crlf}Content-Disposition: form-data; name=\"overwrite\"{crlf}{crlf}true{crlf}"
                f"--{boundary}--{crlf}")
        body = head.encode() + buf.getvalue() + tail.encode()
        req = urllib.request.Request(self.url + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                name = json.loads(r.read())["name"]
        except OSError as exc:
            raise RuntimeError(f"Falha ao enviar a pose ao ComfyUI: {exc}") from exc
        self._uploaded[pose] = name
        return name

    # ---- workflow -----------------------------------------------------------
    def build_prompt(self, prompt: str) -> str:
        head = f"{self.trigger}, " if self.trigger else ""
        return f"{head}{prompt.strip()}, hard pixel edges, limited palette"

    def build_workflow(self, prompt: str, seed: int, pose_image: str | None = None,
                       ref_image: str | None = None, ref_weight: float | None = None) -> dict:
        wf: dict = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.checkpoint}}}
        model, clip = ["1", 0], ["1", 1]
        for i, (name, strength) in enumerate(self.loras):
            nid = f"lora{i}"
            wf[nid] = {"class_type": "LoraLoader", "inputs": {
                "model": model, "clip": clip, "lora_name": name,
                "strength_model": strength, "strength_clip": strength}}
            model, clip = [nid, 0], [nid, 1]
        if ref_image:  # IP-Adapter: mantem o visual do personagem de uma imagem de referencia
            wf["ref_in"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
            wf["ipa_load"] = {"class_type": "IPAdapterUnifiedLoader", "inputs": {
                "model": model, "preset": "PLUS (high strength)"}}
            wf["ipa"] = {"class_type": "IPAdapterAdvanced", "inputs": {
                "model": ["ipa_load", 0], "ipadapter": ["ipa_load", 1], "image": ["ref_in", 0],
                "weight": self.ref_weight if ref_weight is None else ref_weight, "weight_type": "linear",
                "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0, "embeds_scaling": "V only"}}
            model = ["ipa", 0]
        negative = f"{NEGATIVE}, {self.avoid}" if self.avoid else NEGATIVE
        positive, neg = ["3", 0], ["4", 0]
        wf.update({
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": self.build_prompt(prompt)}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negative}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {
                "width": self.native_size, "height": self.native_size, "batch_size": 1}},
            "6": {"class_type": "KSampler", "inputs": {
                "model": model, "seed": seed, "steps": self.steps, "cfg": self.cfg,
                "sampler_name": "euler", "scheduler": "normal",
                "positive": ["cn", 0], "negative": ["cn", 1], "latent_image": ["5", 0], "denoise": 1.0}},
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "openpixel/gen"}},
        })
        if pose_image:
            wf["pose_in"] = {"class_type": "LoadImage", "inputs": {"image": pose_image}}
            wf["cn_model"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": self.controlnet}}
            wf["cn"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": positive, "negative": neg, "control_net": ["cn_model", 0], "image": ["pose_in", 0],
                "strength": self.pose_strength, "start_percent": 0.0, "end_percent": 1.0, "vae": ["1", 2]}}
        else:  # sem pose: o KSampler le direto dos prompts
            wf["6"]["inputs"].update(positive=positive, negative=neg)
        return wf

    # ---- interface do provider ---------------------------------------------
    def generate(self, prompt: str, size: int = 1024, seed: int | None = None, pose: str | None = None,
                 reference: str | None = None, ref_weight: float | None = None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        pose = self.pose if pose is None else pose
        pose_image = self._upload_pose(pose) if pose else None
        ref_image = self._upload_pose(reference) if reference else None  # mesma rotina de upload
        payload = json.dumps({"prompt": self.build_workflow(prompt, seed, pose_image, ref_image, ref_weight), "client_id": uuid.uuid4().hex}).encode()
        pid = self._http("/prompt", payload)["prompt_id"]

        t0 = time.time()
        while time.time() - t0 < self.timeout:
            hist = self._http("/history/" + pid)
            if pid in hist:
                st = hist[pid].get("status", {})
                if st.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI falhou: {json.dumps(st)[:800]}")
                images = [im for o in hist[pid].get("outputs", {}).values() for im in o.get("images", [])]
                if images:
                    it = images[0]
                    q = urllib.parse.urlencode({"filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                                "type": it.get("type", "output")})
                    img = Image.open(io.BytesIO(self._http("/view?" + q, raw=True, timeout=120))).convert("RGB")
                    if self.grid:
                        arr = pixelate(img, grid=self.grid, colors=self.colors, out_size=size)
                    else:
                        if img.size != (size, size):
                            img = img.resize((size, size), Image.LANCZOS)
                        arr = np.asarray(img, dtype=np.uint8)
                    return np.ascontiguousarray(arr), seed
            time.sleep(0.5)
        raise RuntimeError("Tempo maximo de espera do ComfyUI excedido")
