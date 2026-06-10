@echo off
REM One-click Windows build: produces dist\DAS_pro.exe
pip install -r requirements.txt pyinstaller
pyinstaller das_pro.spec --noconfirm
echo.
echo Build finished: dist\DAS_pro.exe
pause
