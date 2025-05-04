@echo off
echo Запускаем мультиактивную торговлю с TD3 агентом...
python main.py --multi_asset --window_size=47 --batch_size=256 --timesteps=100000 --exploration_noise=0.1 --output_dir=results_multi_asset
pause
