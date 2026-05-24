@echo off
cd /d "%~dp0"
ice-simplifier.exe 03543\dress.obj       03543\dress_coarse.obj       450
ice-simplifier.exe 03543\dress_tpose.obj 03543\dress_tpose_coarse.obj 450
pause
