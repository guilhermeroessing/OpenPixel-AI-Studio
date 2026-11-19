> **Documento histórico.** Este guia foi escrito antes dos experimentos de novembro de 2026. O plano de treinar LoRAs próprias foi executado e **descartado** (resultado negativo). O estado atual e o motor de IA recomendado estão no [README](README.md) e em [docs/EXPERIMENTOS.md](docs/EXPERIMENTOS.md).

# Retomada do trabalho: OpenPixel AI Studio

Este arquivo permite continuar, em outra sessão do Claude Code e em outro computador (o da RTX 5050), exatamente de onde o trabalho parou.

## 1. Onde paramos

O projeto está pronto para testes locais em modo mock. Falta treinar o modelo de IA real e integrá-lo à plataforma.

Estado do repositório (`origin` = https://github.com/guilhermeroessing/OpenPixel-AI-Studio, branch `main`), último commit enviado: `c97e0cf`.

| Commit | Conteúdo |
|---|---|
| `43c142e` | Fase 0: estrutura, contratos compartilhados |
| `1bec31f` | backend, gateway MCP, frontend, dataset, scripts `.bat`, testes |
| `52727de` | dataset público (317 assets), categorias, análise de pixels, legendas |
| `c97e0cf` | ingestor privado e `--include-private` (sem dados) |

A última pergunta feita ao usuário, ainda sem resposta no momento deste arquivo:

> Quer que eu faça agora os itens 1 e 6, ou seja, o `metadata.jsonl` com a validação do dataset e o `SDXLProvider`?

## 2. O que já existe

- `shared-contracts/`: schemas Pydantic (`GenerateRequest`, `SpriteResponse`, `MapAsset`, `CompleteMapResponse`) e `pose-templates/` com 8 direções.
- `backend-core/openpixel_core/`: `mock_provider.py` (`SmartMockProvider`), `postprocess.py` (PixelOE com fallback, K-Means CIELAB, remoção de ilhas, padding circular, simetria, dithering Bayer), `service.py`, `api.py` (FastAPI: `/generate`, `/generate8`, `/map`, `/health`, `/vram`, `/dataset`).
- `ai-gateway-mcp/server.py`: FastMCP com 5 tools.
- `frontend-studio/`: React 18, Vite 5, TypeScript, Tailwind 3. O build passa. A aba Edit é só um placeholder.
- `tests/`: 58 testes passando no modo mock.
- `scraper-dataset/`: coletor e `dataset_catalog.db`, com 317 PNGs aprovados e legendas `.txt` versionados.
- `training-local/`: `preprocess.py` (aceita `--include-private`), `train_sdxl_lora.py` (apenas verificado em sintaxe, **nunca executado**) e `ingest_private.py`.
- `setup_windows.bat` e `start_studio.bat`.

## 3. O que NÃO está no Git (precisa ser transferido à mão)

`training-local/private/` contém 3.339 imagens com legenda, organizadas em `images/<categoria>/<pack>__<nome>.png|.txt`, mais `private_catalog.db` e `ingest_report.json`.

Essas imagens vêm das referências pessoais do usuário e **não podem ser redistribuídas**. A decisão do usuário foi manter tudo local, traduzir as imagens em legendas e análise de pixels, e usá-las só para treino. Nunca versione essa pasta. Ela já está no `.gitignore`, junto com `references-local/`.

Transferência (sem usar GitHub, nem repositório privado):

```powershell
# No PC de origem
Compress-Archive -Path "C:\Users\Guilherme\Desktop\openpixel\training-local\private\*" -DestinationPath "$env:USERPROFILE\Desktop\openpixel_private.zip"
Get-FileHash "$env:USERPROFILE\Desktop\openpixel_private.zip" -Algorithm SHA256
```

Leve o zip por pendrive, HD externo ou rede local. No PC da RTX 5050:

```powershell
git clone https://github.com/guilhermeroessing/OpenPixel-AI-Studio
Expand-Archive "$env:USERPROFILE\Desktop\openpixel_private.zip" -DestinationPath ".\OpenPixel-AI-Studio\training-local\private"
Get-FileHash "$env:USERPROFILE\Desktop\openpixel_private.zip" -Algorithm SHA256
```

Confira se o hash final é igual ao do PC de origem. A pasta `references-local/` (755 MB) não precisa ir.

## 4. Preparar o PC da RTX 5050

1. Instale Python e Git, e confirme que `nvidia-smi` funciona (CUDA 12.8 ou superior).
2. Rode `setup_windows.bat`. Se `nvidia-smi` responder, ele deve usar `requirements-gpu.txt` e configurar o `.env` sem `cpu_mock`. **Confira o `.env` gerado** e ajuste `DEVICE_MODE` para o modo CUDA se necessário.
3. Valide o ambiente antes de treinar:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4. Rode os testes (devem passar sem GPU, em modo mock): `python -m pytest tests -q`.

## 5. Plano a seguir, em ordem

1. **Preparar os dados.** Gerar `metadata.jsonl` com uma legenda curta por imagem, um token de gatilho (ex.: `openpixel`) e a categoria. Os `.txt` têm várias linhas (legenda visual, metadados e "Análise de pixels"). Antes disso, **abrir `training-local/train_sdxl_lora.py` e confirmar como ele lê as legendas** (isso ainda não foi verificado). Fazer amostragem de 30 a 50 legendas por categoria.
2. **Definir o desenho do treino.**
   - Spritesheets com padding quadrado em 1024 px desperdiçam resolução. Recomendação: cortar as sheets em frames, ou treinar só com sprites individuais e fundos.
   - Recomendação: 3 LoRAs separadas (personagens; ambiente e paralaxe; efeitos).
3. **Pré-processar** com `training-local/preprocess.py --include-private` (upscale NEAREST para 1024 px). Veja `--help` para os demais argumentos.
4. **Teste curto de VRAM (cerca de 200 passos).** Com 8 GB, SDXL em 1024 px é apertado. Usar gradient checkpointing, Adam de 8 bits, batch 1 com acumulação e, se preciso, 768 px. Se não couber, o fallback é um modelo SD 1.5 de pixel art.
5. **Treino completo.** Gerar amostras com prompts fixos por categoria a cada checkpoint para comparar. O SDXL base (cerca de 7 GB) vem do Hugging Face.
6. **Integrar na plataforma.** Criar um `SDXLProvider` ao lado do `SmartMockProvider`, escolhido por `DEVICE_MODE=cuda`, com imports preguiçosos (diffusers só carrega no modo GPU). A saída passa pelo pipeline existente (PixelOE, K-Means CIELAB, padding circular), então `/generate`, o gateway MCP e o frontend não mudam. Testar em mock no PC fraco. ControlNet OpenPose para as 8 direções fica para depois.

## 6. Pendências com o usuário

- Rodar no próprio terminal: `claude mcp add --transport http spritefusion https://www.spritefusion.com/mcp` (a CLI `claude` não estava no PATH da sessão anterior).
- Wikimedia Commons retorna 403 por causa do User-Agent genérico. Falta o usuário escolher o contato a usar (URL do repositório ou ignorar essa fonte).
- Decidir se vale criar um ingestor `manual-inbox` para novos packs do itch.io.

## 7. Problemas conhecidos nos dados

- As contagens de frames nas legendas são, na maior parte, estimativas visuais e muitas vezes diferem da análise medida.
- Ícones de 16 a 32 px têm legendas genéricas ou incertas, e alguns nomes de criaturas foram inferidos.
- As categorias são heurísticas. No dataset público `terrain_parallax` tem 0 itens, e no privado tem 551.
- O PDF de licença de um dos packs não pôde ser lido, então essa licença está sem verificação.

## 8. Cuidados de ambiente (Windows)

- Caminhos longos (MAX_PATH) quebram a extração de zips aninhados. Use destinos curtos.
- Scripts que precisam do Pillow do site-packages do usuário: `python -I` o ignora. Use `python -E -P`.
- Regras que o usuário definiu: nunca versionar `references-local/` e `training-local/private/`, e nunca redistribuir as imagens privadas. Kenney, FreeGameAssets, Adobe Stock, pixelrepo e automação do itch.io foram descartados por robots.txt ou termos de uso.

## 9. Prompt para colar na nova sessão

```text
Estou retomando o projeto OpenPixel AI Studio neste computador (RTX 5050 8GB). Leia RETOMADA_SESSAO.md na raiz do repositório, além de openpixel_ai_studio_architecture.md e implementation_roadmap.md, e siga exatamente o plano da seção 5.

Antes de qualquer coisa: confirme que a pasta training-local/private/ foi copiada e que o hash do zip confere, rode nvidia-smi e o teste de torch.cuda.is_available(), verifique o .env gerado por setup_windows.bat e rode os testes (pytest tests -q).

Depois comece pelo item 1: confira como training-local/train_sdxl_lora.py lê as legendas, gere o metadata.jsonl com validação do dataset (legenda curta, token de gatilho, categoria) e mostre uma amostra de 30 a 50 legendas por categoria para eu aprovar. Em seguida faça o item 6 (SDXLProvider com imports preguiçosos, escolhido por DEVICE_MODE=cuda, com testes em modo mock).

Regras: nunca versione references-local/ nem training-local/private/, não redistribua as imagens, não faça commit ou push sem eu pedir, e pare para me perguntar antes de iniciar o treino completo.
```
