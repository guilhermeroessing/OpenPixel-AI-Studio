# OpenPixel AI Studio - System Architecture & Guidelines

## 1. Visão Geral e Identidade do Projeto
**OpenPixel AI Studio** é uma plataforma Open Source completa para criação de **Personagens, Assets Isolados, Tilesets e Mapas Completos em Pixel Art 2D** com IA local.
Esta arquitetura foi desenhada para suportar desenvolvimento distribuído: codificação, scraping e testes visuais em um PC sem placa de vídeo dedicada (Mock Mode / GitHub), e execução profunda/treinamento em uma máquina local com NVIDIA RTX 5050 (8GB VRAM / CUDA 12.8+).

### 1.1 Identidade Visual (Frontend)
A interface adotará um design moderno:
* **Paleta:** Dark Mode baseado em `Zinc/Slate` (`bg-zinc-950`).
* **Tipografia e Ícones:** Fonte `Inter` e ícones SVG (`lucide-react`).
* **Layout Modular:** Sidebar Esquerda fixa (controles, pílulas de tamanho 16-128, monitor VRAM, aba Dataset), Painel Flutuante Inferior (Animate, Edit, Map Blueprint) e Popover de Settings no topo (CRT Mode, Checkerboard, Pixel Perfect).
* **Canvas Central:** Fundo xadrez via CSS com `image-rendering: pixelated` estrito.

---

## 2. Estratégia de Hardware Seguro (Dev/Mock vs. GPU)
Para evitar travamentos no PC de desenvolvimento, o sistema divide suas dependências e motores:
* **Dependências Duplas:** O projeto deve ter `requirements-dev.txt` (somente FastAPI, OpenCV, NumPy, Pillow, PixelOE, FastMCP, sem PyTorch/CUDA) e `requirements-gpu.txt` (PyTorch cu128, diffusers, xformers).
* **Auto-Detect no `setup_windows.bat`:** O script verificará a presença do `nvidia-smi`. Se falhar, definirá `DEVICE_MODE=cpu_mock` no `.env`.
* **Motor `SmartMockProvider` (PC Fraco):** Quando em mock, em vez de carregar modelos pesados (SDXL), o backend vai gerar dinamicamente formas geométricas com ruído perlin e degradês. O pipeline matemático (downscale `PixelOE`, quantização K-Means CIELAB e dithering) processará esse rascunho gerando uma *pixel art de teste* real na UI, permitindo validar todo o software sem GPU.

---

## 3. Pipeline de Web Scraping e Banco SQLite (`/scraper-dataset`)
O Claude Code deve executar o scraper para montar o dataset, mas com **travas de segurança para o GitHub**:
* **Limite de Coleta (`--limit 50`):** Para permitir que as imagens fiquem salvas no repositório GitHub para uso futuro no PC principal, limite a extração inicial a no máximo 50 assets aprovados por fonte.
* **Armazenamento no Git:** A pasta `/scraper-dataset/processed/` e o banco `dataset_catalog.db` **NÃO** devem entrar no `.gitignore` nesta fase.
* **7 Fontes Abertas (Sem Login):** OpenGameArt, Kenney, Dungeon Crawl Stone Soup (GitHub repo `rltiles`), Liberated Pixel Cup (GitHub repo), FreeGameAssets, Lospec e Wikimedia Commons API.
* **Filtro Matemático:** O fatiador OpenCV deve reduzir imagens via GCD (escala 1:1), descartar mixels/anti-aliasing, separar `isolated_sprite` de `full_tile_scene` e catalogar paletas.

---

## 4. Motor Backend e Gerador de Mapas (`/backend-core`)
* **Consistência:** Implementar *Circular Padding* (para cenários sem emendas), Simetria Bilateral, K-Means CIELAB (trava de subpaletas em loops de animação) e Inpainting com trava de pixels não alterados.
* **Decomposição de Mapas:** O prompt de cenário gera um JSON com `master_palette`, `asset_manifest` (lote de assets a serem gerados) e `tilemap_layers`. O sistema gera os tiles, props e o arquivo **`MAP_ASSEMBLY_MANUAL.md`** explicando como montar na Godot/Unity.

---

## 5. Gateway MCP (`/ai-gateway-mcp`)
Servidor `FastMCP` rodando local com as tools: `generate_pixel_art`, `generate_8_directions`, `animate_asset`, `generate_complete_map` (retorna o pacote de assets e o manual) e `export_engine_bundle`.

---

## 6. Sincronização Obrigatória com GitHub (Claude Code Instructions)
O agente deve inicializar o controle de versão e enviá-lo ao GitHub remotamente.
1. `git init`
2. Configure o `.gitignore` para ignorar `node_modules/`, `__pycache__/`, `.venv/` e `.env`, **MAS GARANTA** que a pasta `/scraper-dataset/` inteira seja comitada.
3. Se a CLI do GitHub (`gh`) estiver disponível, execute: `gh repo create OpenPixel-AI-Studio --public --source=. --remote=origin --push`.
4. Se o `gh` não estiver disponível, pare e peça ao usuário a URL do repositório remoto criado manualmente, adicione com `git remote add origin <URL>` e execute o `git push -u origin main`.