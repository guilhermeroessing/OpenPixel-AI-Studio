@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"

echo === OpenPixel AI Studio - setup Windows ===

rem 1. Detecta GPU NVIDIA via nvidia-smi
nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "MODE=cpu_mock"
    set "REQ=requirements-dev.txt"
    echo [INFO] nvidia-smi nao encontrado. Modo: cpu_mock ^(sem PyTorch/CUDA^).
) else (
    set "MODE=cuda"
    set "REQ=requirements-gpu.txt"
    echo [INFO] GPU NVIDIA detectada. Modo: cuda.
)

rem 2. Garante .env e grava DEVICE_MODE (preserva as demais variaveis)
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
    ) else (
        type nul > ".env"
    )
)
findstr /v /b /c:"DEVICE_MODE=" ".env" > ".env.tmp"
echo DEVICE_MODE=%MODE%>> ".env.tmp"
move /y ".env.tmp" ".env" >nul
echo [INFO] .env atualizado: DEVICE_MODE=%MODE%

rem 3. Python e ambiente virtual
where python >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH. Instale Python 3.11+ marcando "Add to PATH".
    goto :fail
)
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Criando .venv...
    python -m venv .venv
    if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

rem 4. Dependencias Python conforme o modo
echo [INFO] Instalando %REQ% ...
python -m pip install -r "%REQ%"
if errorlevel 1 goto :fail

rem 5. Dependencias do frontend
where npm >nul 2>&1
if errorlevel 1 (
    echo [AVISO] npm nao encontrado. Instale Node.js LTS e rode este setup de novo.
    goto :fail
)
if not exist "frontend-studio\package.json" (
    echo [AVISO] frontend-studio\package.json nao encontrado. npm install pulado.
) else (
    pushd "frontend-studio"
    call npm install
    if errorlevel 1 (
        popd
        goto :fail
    )
    popd
)

echo === Setup concluido. DEVICE_MODE=%MODE% ===
echo Rode start_studio.bat para subir o ambiente.
pause
exit /b 0

:fail
echo [ERRO] Setup interrompido. Veja a mensagem acima.
pause
exit /b 1
