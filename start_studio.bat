@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERRO] .venv nao encontrado. Rode setup_windows.bat primeiro.
    pause
    exit /b 1
)

rem Backend FastAPI em janela propria
echo [INFO] Subindo backend em http://127.0.0.1:8000 ...
start "OpenPixel API :8000" /D "%ROOT%" cmd /k call ".venv\Scripts\activate.bat" ^& python -m uvicorn openpixel_core.api:app --app-dir backend-core --host 127.0.0.1 --port 8000 --reload

rem Frontend Vite em janela propria
if exist "frontend-studio\package.json" (
    echo [INFO] Subindo frontend com npm run dev ...
    start "OpenPixel Frontend" /D "%ROOT%frontend-studio" cmd /k npm run dev
) else (
    echo [AVISO] frontend-studio\package.json nao encontrado. Frontend pulado.
)

rem Gateway MCP (opcional)
choice /c SN /m "Subir tambem o gateway MCP? S/N"
if errorlevel 2 goto :done
if not exist "ai-gateway-mcp\server.py" (
    echo [AVISO] ai-gateway-mcp\server.py nao encontrado. Gateway pulado.
    goto :done
)
start "OpenPixel MCP Gateway" /D "%ROOT%" cmd /k call ".venv\Scripts\activate.bat" ^& python ai-gateway-mcp\server.py

:done
echo [INFO] Processos iniciados em janelas separadas. Feche as janelas para parar.
exit /b 0
