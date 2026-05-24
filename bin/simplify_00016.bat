@echo off
cd /d "%~dp0"
ice-simplifier.exe 00016\shirt.obj 00016\shirt_coarse.obj 400
ice-simplifier.exe 00016\shirt_tpose.obj 00016\shirt_tpose_coarse.obj 400
pause
