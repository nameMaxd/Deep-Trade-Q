@echo off
chcp 65001 >nul
cd /d %~dp0
python train.py data/GOOG_2010-2024-06.csv --strategy=td3 --window-size=47 --td3-save-name=td3_model --td3-timesteps=100000 --td3-noise-sigma=1.0 --debug
pause
