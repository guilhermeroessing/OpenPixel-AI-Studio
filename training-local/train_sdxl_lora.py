"""Fine-tuning base de LoRA para o UNet do SDXL, alvo: RTX 5050 8GB (CUDA 12.8+).

Espera pares PNG + legenda TXT em training-local/data/processed_1024
(gerado por preprocess.py). As latentes da VAE e os embeddings de texto sao
pre-calculados uma vez, e os modelos de VAE/texto sao descarregados antes do
treino para caber na VRAM. Apenas os adaptadores LoRA sao treinados.

Uso (a partir da raiz do projeto, com .venv ativo):
    python training-local/train_sdxl_lora.py --steps 1500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
except ImportError as exc:
    sys.exit(
        f"[ERRO] Dependencia de treino ausente: {exc.name}\n"
        "Ative o ambiente e instale as dependencias de GPU:\n"
        "    .venv\\Scripts\\activate\n"
        "    pip install -r requirements-gpu.txt\n"
    )

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDXL LoRA fine-tuning (base).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", type=Path, default=ROOT / "training-local" / "data" / "processed_1024")
    p.add_argument("--output-dir", type=Path, default=ROOT / "training-local" / "output" / "sdxl-lora")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000, help="passos de otimizacao")
    p.add_argument("--grad-accum", type=int, default=4, help="micro-batches por passo (batch efetivo)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16, help="rank da LoRA")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xformers", action="store_true", help="usa atencao xformers se instalado")
    return p.parse_args()


def load_pairs(folder: Path) -> list[tuple[Path, str]]:
    pairs = []
    for img in sorted(folder.glob("*.png")):
        cap = img.with_suffix(".txt")
        if cap.exists():
            text = cap.read_text(encoding="utf-8").strip()
            if text:
                pairs.append((img, text))
    return pairs


def load_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1)
    return tensor.unsqueeze(0)  # (1, 3, H, W) em [-1, 1]


@torch.no_grad()
def encode_caption(pipe, caption: str, device: torch.device):
    hidden = []
    pooled = None
    for tok, enc in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)):
        ids = tok(
            caption,
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        out = enc(ids.to(device), output_hidden_states=True)
        pooled = out[0]  # text_embeds do CLIP com projecao (encoder 2)
        hidden.append(out.hidden_states[-2])
    return torch.cat(hidden, dim=-1).cpu().half(), pooled.cpu().half()


@torch.no_grad()
def precompute(pipe, pairs, device: torch.device):
    """Calcula latentes e embeddings, depois libera VAE e text encoders."""
    latents, embeds, pooleds = [], [], []

    pipe.vae.to(device, dtype=torch.float32)  # VAE em fp32 evita NaN do SDXL em fp16
    pipe.vae.enable_tiling()
    for img, _ in pairs:
        px = load_pixels(img).to(device, torch.float32)
        lat = pipe.vae.encode(px).latent_dist.sample() * pipe.vae.config.scaling_factor
        latents.append(lat.cpu().half())
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"[INFO] {len(latents)} latentes calculadas")

    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    for _, cap in pairs:
        e, p = encode_caption(pipe, cap, device)
        embeds.append(e)
        pooleds.append(p)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    print(f"[INFO] {len(embeds)} embeddings de texto calculados")

    return latents, embeds, pooleds


def save_lora(pipe, unet, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=state,
        weight_name=f"{name}.safetensors",
    )
    print(f"[OK] LoRA salvo em {out_dir / (name + '.safetensors')}")


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit(
            "[ERRO] CUDA indisponivel. Este script exige GPU NVIDIA.\n"
            "Verifique 'nvidia-smi' e se o torch instalado e o cu128 (requirements-gpu.txt)."
        )

    pairs = load_pairs(args.data_dir) if args.data_dir.is_dir() else []
    if not pairs:
        sys.exit(f"[ERRO] Nenhum par PNG + TXT em {args.data_dir}. Rode training-local/preprocess.py antes.")

    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[INFO] {len(pairs)} pares imagem/legenda | modelo base: {args.model}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True
    )
    latents, embeds, pooleds = precompute(pipe, pairs, device)

    # Congela o UNet e injeta LoRA nas projecoes de atencao
    unet = pipe.unet
    unet.to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # xformers opcional
            print(f"[AVISO] xformers indisponivel, usando atencao padrao: {exc}")
    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    cast_training_params(unet, dtype=torch.float32)  # so os pesos LoRA em fp32
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[INFO] parametros LoRA treinaveis: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    time_ids = torch.tensor(
        [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
        device=device,
        dtype=torch.float16,
    )

    unet.train()
    for step in range(1, args.steps + 1):
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            i = random.randrange(len(latents))
            lat = latents[i].to(device)
            emb = embeds[i].to(device)
            pool = pooleds[i].to(device)

            noise = torch.randn_like(lat)
            t = torch.randint(
                0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device
            ).long()
            noisy = scheduler.add_noise(lat, noise, t)

            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(
                    noisy,
                    t,
                    encoder_hidden_states=emb,
                    added_cond_kwargs={"text_embeds": pool, "time_ids": time_ids},
                ).sample
                loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum

            scaler.scale(loss).backward()
            loss_sum += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 50 == 0:
            print(f"[step {step}/{args.steps}] loss={loss_sum:.4f}")
        if args.save_every and step % args.save_every == 0:
            save_lora(pipe, unet, args.output_dir, f"checkpoint-{step}")

    save_lora(pipe, unet, args.output_dir, "openpixel_sdxl_lora")
    return 0


if __name__ == "__main__":
    sys.exit(main())
