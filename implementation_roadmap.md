# Plano de Execução Multi-Fase (Para o Claude Code)

Este documento define a ordem estrita de execução para o **OpenPixel AI Studio**. Você (agente) deve implementar as lógicas visando o **modo de desenvolvimento em PC fraco**, mantendo o código pronto para a RTX 5050.

## Fase 0: Fundações, Git e Contratos Compartilhados

1. **Repositório e GitHub:**
   * Inicialize o repositório Git e crie a estrutura: `shared-contracts`, `scraper-dataset`, `training-local`, `backend-core`, `ai-gateway-mcp`, `frontend-studio`, e `tests`.
   * Crie o `.gitignore` poupando a pasta de imagens do scraper (para que o usuário possa cloná-las no PC mais forte depois).
   * Tente criar o repositório remotamente via `gh repo create`. Se falhar, solicite ao usuário a URL do repositório antes de prosseguir.

2. **Ambiente e Mock:**
   * Crie `requirements-dev.txt` (sem PyTorch/CUDA) e `requirements-gpu.txt`.
   * Implemente o `SmartMockProvider`: um gerador de matrizes NumPy com ruídos que simula o output de IA, para testar a matemática de PixelOE e CIELAB na CPU.

3. **Contratos (`/shared-contracts`):**
   * Pydantic Schemas: `GenerateRequest`, `SpriteResponse`, `CompleteMapResponse`.
   * Templates JSON OpenPose em `pose-templates/`.

## Fase 1: Coleta Segura para GitHub (`/scraper-dataset`)

**Ação Obrigatória:** Escreva e execute o coletor assíncrono para extrair, fatiar e classificar assets das 7 fontes abertas listadas no `CLAUDE.md`.
* Aplique estritamente o limite de **50 assets salvos por fonte** para não exceder o limite de upload do GitHub.
* Processe tudo pelo filtro matemático (OpenCV, GCD para escala 1:1, descarte de mixels/blur) e salve no SQLite (`dataset_catalog.db`).
* Gere legendas textuais descritivas `.txt` parciais para esses assets simulando um processo multimodal.
* Dê um `git add .` e `git commit -m "feat: dataset populado e filtrado"` para salvar essas imagens no histórico.

## Fase 2: Scripts de Treinamento e Ambientes Windows

* Implemente `setup_windows.bat`. Ele deve rodar `nvidia-smi`; se falhar, deve configurar `.env` com `DEVICE_MODE=cpu_mock` e instalar o `requirements-dev.txt`.
* Crie o pipeline de pré-processamento (Upscale via `Image.NEAREST` para 1024px) e o script base de fine-tuning SDXL (deixando-o pronto para ser rodado na máquina RTX).

## Fase 3: Motor Pós-IA e MapBuilder (`/backend-core`)

* **Pós-processamento (CPU friendly):** Programe e ative a lógica do `PixelOE`, remoção de ilhas (1-2px) via OpenCV, quantização travada em subpaletas e *Circular Padding* (para cenários).
* **Gerador de Mapas:** Programe a decomposição de prompts de mapas, criação da `master_palette`, gerador do JSON para Godot/Unity, renderização da montagem (`composite_map.png`) e a escrita do `MAP_ASSEMBLY_MANUAL.md`. Tudo operando no `SmartMockProvider` por padrão.

## Fase 4: Gateway MCP e UI React (`/ai-gateway-mcp` e `/frontend-studio`)

* Configure as tools do `FastMCP` (pixel_art, mapas, 8_directions).
* Construa a UI em React/Vite com `Tailwind Zinc`, Canvas xadrez, Sidebar de pílulas e Painel Flutuante. O monitor de VRAM e o dataset (com os dados reais do SQLite) devem ser renderizados.

## Fase 5: Validação e Commit Final no GitHub

* Escreva os testes no `pytest` para validar o fluxo MCP (`generate_complete_map` retornando o manual sem quebrar).
* Dê o push final de todo o ecossistema (backend, frontend, testes, banco de dados SQLite e os limitados PNGs do scraper) para o repositório remoto: `git push origin main`.
* Informe ao usuário que o projeto está pronto para testes locais no PC dev via `start_studio.bat` e empacotado no GitHub para deploy na RTX 5050.