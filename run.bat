@echo off
cd /d "%~dp0"
rem Rebuild the environment if it is missing or was moved from another folder
set "LOC="
if exist .venv\location.txt set /p LOC=<.venv\location.txt
if exist .venv if /i not "%LOC%"=="%~dp0" (
  echo App folder moved - rebuilding its environment, takes a few minutes...
  rmdir /s /q .venv
)
if not exist .venv (
  echo Setting up Sorter - first run takes a few minutes...
  python -m venv .venv || (echo Python not found. Install Python 3.12 from python.org and tick "Add to PATH". & pause & exit /b)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || .venv\Scripts\python -m pip install torch
  echo %~dp0>.venv\location.txt
)
fc /b requirements.txt .venv\req.txt >nul 2>&1 || (
  .venv\Scripts\python -m pip install -r requirements.txt && copy /y requirements.txt .venv\req.txt >nul
)
if not exist .venv\shortcut.done (
  powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Sorter.lnk');$s.TargetPath='%~dp0run.bat';$s.WorkingDirectory='%~dp0';$s.IconLocation='%~dp0sorter.ico';$s.WindowStyle=7;$s.Save()" && echo.>.venv\shortcut.done
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0photo_sorter.py"
