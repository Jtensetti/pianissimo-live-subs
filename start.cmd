@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo Skapar lokal Python-miljo...
    py -3.11 -m venv .venv
    if errorlevel 1 goto failed
)
if not exist ".venv\pianissimo-ready.txt" (
    echo Forsta starten: installerar programmets beroenden...
    ".venv\Scripts\python.exe" -m pip install -r backend\requirements.txt
    if errorlevel 1 goto failed
    echo ready>".venv\pianissimo-ready.txt"
)
".venv\Scripts\python.exe" run.py %*
exit /b %errorlevel%
:failed
echo.
echo Starten misslyckades. Kontrollera att Python 3.11 med Python Launcher ar installerat.
echo Forsta installationen behover internet.
exit /b 1
