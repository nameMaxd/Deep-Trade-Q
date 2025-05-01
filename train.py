"""
Script for training Stock Trading Bot.

Usage:
  train.py <stock> [<dummy>] [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] [--model-type=<model-type>]
    [--target-update=<target-update>] [--td3-timesteps=<td3-timesteps>]
    [--td3-noise-sigma=<td3-noise-sigma>] [--td3-save-name=<td3-save-name>]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 20]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
  --model-type=<model-type>         Model type: 'dense' (default) or 'lstm'.
  --target-update=<target-update>    Target network update frequency (episodes). [default: 100]
  --td3-timesteps=<td3-timesteps>    Total timesteps for TD3 training. [default: 100000]
  --td3-noise-sigma=<td3-noise-sigma>  # Sigma для TD3 action noise (по умолчанию 1.0)
  --td3-save-name=<td3-save-name>    Filename to save TD3 model (no extension). [default: td3_model]
  --force-cpu                       Force using CPU instead of GPU for TensorFlow and PyTorch.

"""

import logging
import coloredlogs
import os
import sys
import traceback
import numpy as np
import pandas as pd

# Включаем отладочный вывод
print("=== Запуск скрипта train.py ===")
print(f"Аргументы: {sys.argv}")
print(f"Текущая директория: {os.getcwd()}")

# Проверяем наличие файла данных
if len(sys.argv) > 1:
    data_file = sys.argv[1]
    if os.path.exists(data_file):
        print(f"Файл данных найден: {data_file}")
    else:
        data_path = os.path.join('data', data_file)
        if os.path.exists(data_path):
            print(f"Файл данных найден в подкаталоге data: {data_path}")
        else:
            print(f"ОШИБКА: Файл данных не найден ни в {data_file}, ни в {data_path}")

# Парсим аргументы перед импортом TensorFlow и PyTorch
force_cpu = '--force-cpu' in sys.argv
if force_cpu:
    print("Принудительно используем CPU вместо GPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем GPU для TensorFlow

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    
    # Применяем mixed_precision только если не force_cpu
    if not force_cpu:
        mixed_precision.set_global_policy('mixed_float16')
    print("TensorFlow успешно загружен")
except Exception as e:
    print(f"Ошибка при загрузке TensorFlow: {e}")
    traceback.print_exc()

try:
    import io
    import re
    import numpy as np
    import pandas as pd
    import logging
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    # Делаем torch доступным глобально
    global torch
    try:
        import torch
        # Настройка PyTorch для использования CPU
        if force_cpu:
            torch.set_num_threads(os.cpu_count())  # Максимально эффективно используем CPU
        print("PyTorch успешно загружен")
    except ImportError:
        print("Ошибка импорта PyTorch")
    
    # Импортируем модули Stable-Baselines3
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        print("Stable-Baselines3 успешно загружен")
    except ImportError as e:
        print(f"Ошибка импорта Stable-Baselines3: {e}")
        sys.exit(1)
    
    # Загружаем модули проекта
    from trading_bot.env import TradingEnv
    from trading_bot.ops import get_state
    from trading_bot.methods import train_model, evaluate_model
    from trading_bot.agent import Agent  # Добавляем импорт класса Agent
    print("Модули проекта успешно загружены")
except Exception as e:
    print(f"Ошибка при загрузке библиотек: {e}")
    traceback.print_exc()
    sys.exit(1)
try:
    from trading_bot.utils import (
        WINDOW_SIZE,
        minmax_normalize,
        format_currency,
        format_position,
        show_train_result,
        switch_k_backend_device
    )
    print("Утилиты успешно загружены")
    
    # Configure multi-core usage - оптимизируем для процессора
    import multiprocessing
    num_physical_cores = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_physical_cores)
    # Используем все физические ядра для максимальной производительности
    tf.config.threading.set_inter_op_parallelism_threads(num_physical_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_physical_cores)
    print(f"Многопоточность настроена на {num_physical_cores} ядер")
except Exception as e:
    print(f"Ошибка при загрузке утилит или настройке многопоточности: {e}")
    traceback.print_exc()
    sys.exit(1)

def save_window_size(model_path, window_size):
    base = os.path.splitext(model_path)[0]
    fname = base + ".window_size.txt"
    with open(fname, "w") as f:
        f.write(str(window_size))

def main(stock, window_size=47, batch_size=32, ep_count=50, strategy="t-dqn", model_name=None, pretrained=False, debug=False, target_update=100, td3_timesteps=100000, td3_noise_sigma=1.0, td3_save_name='td3_model'):
    # Используем глобальные библиотеки
    global pd, np
    # Импортируем torch непосредственно в функции main
    import torch
    
    print("=== ЗАПУСК ФУНКЦИИ MAIN ===")
    print("Полученные параметры:")
    print(f"  stock={stock}")
    print(f"  window_size={window_size}")
    if strategy == "td3":
        print(f"  strategy={strategy}")
        print(f"  td3_timesteps={td3_timesteps}")

    # Настраиваем логирование в файл
    logging.basicConfig(filename="train_finetune.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("Лог обучения будет писаться в train_finetune.log")

    # ЯВНОЕ разделение train/val: никаких дат, никаких пересечений!
    train_path = stock  # путь к train-файлу передаётся аргументом
    val_path = os.path.join(os.path.dirname(__file__), 'data', 'GOOG_2024-07_2025-04.csv')
    # train
    train_df = pd.read_csv(train_path)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    raw_train_prices = train_df["Adj Close"].values
    raw_train_volumes = train_df["Volume"].values
    price_norm = list(minmax_normalize(raw_train_prices))
    vol_norm = list(minmax_normalize(raw_train_volumes))
    train_data = list(zip(price_norm, vol_norm))
    # val
    val_df = pd.read_csv(val_path)
    val_df["Date"] = pd.to_datetime(val_df["Date"])
    val_df = val_df.sort_values("Date").reset_index(drop=True)
    raw_val_prices = val_df["Adj Close"].values
    raw_val_volumes = val_df["Volume"].values
    val_price = list(minmax_normalize(raw_val_prices))
    val_vol = list(minmax_normalize(raw_val_volumes))
    val_data = list(zip(val_price, val_vol))
    print(f"Train: {len(train_df)} days")
    print(f"train_data: min={np.min(train_data):.2f}, max={np.max(train_data):.2f}, mean={np.mean(train_data):.2f}")
    # Принудительно создать лог-файл
    with open("train_finetune.log", "a") as f: f.write("=== Training started ===\n")
    # TD3 strategy: train with Stable-Baselines3 and visualize via VisualizeCallback
    if strategy == 'td3':
        # Setup logging directories
        plots_dir = os.path.join(os.getcwd(), 'plots')
        monitor_dir = os.path.join(plots_dir, 'monitor')
        tb_dir = os.path.join(plots_dir, 'tensorboard')
        os.makedirs(monitor_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
        from trading_bot.env import TradingEnv
        from trading_bot.visualize_callback import VisualizeCallback

        # load and clean CSV
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        raw_lines = []
        with open(stock, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Date,'):
                    raw_lines.append(line)
                else:
                    m = pattern.search(line)
                    if m:
                        raw_lines.append(line[m.start():])
        df = pd.read_csv(io.StringIO('\n'.join(raw_lines)))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        # split train/val
        train_df = df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2024-01-01')]
        val_df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-06-30')]
        train_prices = train_df['Adj Close'].values
        train_vols = train_df['Volume'].values
        val_prices = val_df['Adj Close'].values
        val_vols = val_df['Volume'].values
        
        # Загружаем дополнительный файл для валидации
        val_file = 'data/GOOG_2024-07_2025-04.csv'
        if os.path.exists(val_file):
            print(f'Загрузка дополнительных данных для валидации: {val_file}')
            # Загружаем и очищаем CSV валидации
            val_raw_lines = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('Date,'):
                        val_raw_lines.append(line)
                    else:
                        m = pattern.search(line)
                        if m:
                            val_raw_lines.append(line[m.start():])
            val_ext_df = pd.read_csv(io.StringIO('\n'.join(val_raw_lines)))
            val_ext_df['Date'] = pd.to_datetime(val_ext_df['Date'])
            val_ext_df = val_ext_df.sort_values('Date').reset_index(drop=True)
            
            # Используем эти данные для валидации
            val_prices = val_ext_df['Adj Close'].values
            val_vols = val_ext_df['Volume'].values
            print(f'Загружены данные для валидации: {len(val_prices)} точек')

        # Создаем окружения для обучения и валидации
        train_env_raw = TradingEnv(train_prices, train_vols, window_size)
        val_env_raw = TradingEnv(val_prices, val_vols, window_size)
        
        # Сбрасываем окружения перед использованием
        train_env_raw.reset()
        val_env_raw.reset()
        
        # Обертываем в Monitor для логирования наград
        train_env_monitored = Monitor(train_env_raw, monitor_dir)
        train_env_monitored.reset()
        
        # Дополнительно обернем в DummyVecEnv для совместимости с Stable-Baselines3
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([lambda: train_env_monitored])
        val_env = DummyVecEnv([lambda: val_env_raw])
        
        # Сбрасываем векторизованные окружения
        train_env.reset()
        val_env.reset()
        
        n_actions = train_env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
        
        # Initialize TD3 с параметрами для стабильного обучения и предотвращения переобучения
        model = TD3(
            'MlpPolicy', 
            train_env, 
            action_noise=action_noise, 
            verbose=1,
            tensorboard_log=tb_dir,
            learning_rate=0.0003,  # Еще больше уменьшаем скорость обучения для стабильности
            buffer_size=50000,    # Увеличиваем буфер для лучшего обучения
            batch_size=256,       # Увеличиваем размер батча для стабильности
            train_freq=(5, 'step'),  # Обучаем каждые 5 шагов для стабильности
            gradient_steps=5,     # Меньше шагов градиентного спуска для стабильности
            learning_starts=1000,  # Начинаем обучение после накопления достаточного количества опыта
            tau=0.005,            # Медленнее обновляем целевые сети для стабильности
            policy_delay=2,       # Реже обновляем политику для стабильности
            target_policy_noise=0.2,  # Уменьшаем шум для стабильности
            target_noise_clip=0.5,    # Ограничиваем шум для стабильности
            policy_kwargs={
                'activation_fn': torch.nn.ReLU,  # Используем ReLU для лучшего обучения
                'net_arch': {
                    'pi': [128, 64, 32],  # Более широкая архитектура политики (actor)
                    'qf': [256, 128, 64]  # Более широкая архитектура Q-функции (critic)
                }
            }
        )
        
        # Передаем оригинальные окружения для визуализации
        visual_cb = VisualizeCallback(train_env, val_env, model, train_env_raw, val_env_raw, total_timesteps=td3_timesteps)
        
        # Запускаем обучение в несколько эпизодов
        print("Запуск обучения TD3 в несколько эпизодов...")
        
        # Увеличиваем количество эпизодов обучения для лучшего результата
        num_episodes = 10  # Увеличиваем с 5 до 10 эпизодов
        timesteps_per_episode = td3_timesteps // num_episodes
        
        for episode in range(1, num_episodes + 1):
            print(f"\n=== Эпизод {episode}/{num_episodes} ===")
            model.learn(total_timesteps=timesteps_per_episode, callback=visual_cb, reset_num_timesteps=False)
        model.save(f"{td3_save_name}")
        
        # Сохраняем график наград из логов Monitor
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            
            # Пытаемся прочитать логи из Monitor
            monitor_file = os.path.join(monitor_dir, 'monitor.csv')
            if os.path.exists(monitor_file):
                # Читаем CSV с наградами
                df = pd.read_csv(monitor_file, skiprows=1)  # Пропускаем строку с комментарием
                
                if 'r' in df.columns and len(df) > 0:
                    # Преобразуем данные в numpy массивы для избежания ошибки с индексированием
                    x = np.arange(len(df))
                    y = df['r'].values  # Используем .values для получения numpy массива
                    
                    plt.figure(figsize=(12, 4))
                    plt.plot(x, y, label='Episode Reward')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title(f'TD3 Training Rewards - {stock}')
                    plt.legend()
                    plt.savefig(os.path.join(plots_dir, 'rewards.png'))
                    plt.close()
                else:
                    print("Нет данных о наградах в логах Monitor")
            else:
                print(f"Файл логов {monitor_file} не найден")
        except Exception as e:
            print(f"Ошибка при построении графика наград: {e}")
        return
    # если выбран TD3, запускаем Stable-Baselines3 TD3
    if strategy.lower() == 'td3':
        # --- ВОССТАНОВЛЕНИЕ: подготовка pretrained/model_name и финальный инференс ---
        import glob
        if pretrained:
            # find latest .h5 in models/ if no specific file
            if not model_name or not model_name.endswith('.h5'):
                h5_files = glob.glob(os.path.join('models', '*.h5'))
                if not h5_files:
                    logging.warning('No pretrained .h5 found in models/, training from scratch.')
                    pretrained = False
                    model_name = None
                else:
                    h5_files.sort(key=os.path.getmtime, reverse=True)
                    model_name = os.path.basename(h5_files[0])
            else:
                # keep only filename
                model_name = os.path.basename(model_name)

        # Initialize agent with dynamic window_size (state_size computed internally)
        agent = Agent(window_size, strategy=strategy, reset_every=target_update, pretrained=pretrained, model_name=model_name)
        agent.model_type = model_type
        if model_type == 'lstm' and model_name and not model_name.endswith('_LSTM'):
            model_name += '_LSTM'
            agent.model_name = model_name

        best_val_profit = None
        best_val_epoch = None
        no_improve = 0
        strategies = strategy.split(",")
        for strat in strategies:
            agent.strategy = strat
        # --- КОНЕЦ ВОССТАНОВЛЕНИЯ ---

        from trading_bot.env import TradingEnv
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
        import torch

        # Progress bar for TD3
        class TqdmCallback(BaseCallback):
            def __init__(self, total_timesteps):
                super().__init__()
                self.total_timesteps = total_timesteps
            def _on_training_start(self):
                self.pbar = tqdm(total=self.total_timesteps, desc='TD3 Training')
            def _on_step(self) -> bool:
                # update progress
                self.pbar.n = self.num_timesteps
                self.pbar.refresh()
                return True
            def _on_training_end(self):
                self.pbar.close()
        # Callback: logs train & val profit, Sharpe, avg reward, trades; saves best; early stop
        class EvalCallbackTD3(BaseCallback):
            def __init__(self, train_env, eval_env, eval_freq, n_eval_episodes, patience, save_path):
                super().__init__()
                self.train_env = train_env
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes
                self.patience = patience
                self.best_val_profit = -float('inf')
                self.no_improve = 0
                self.save_path = save_path

            def _on_step(self) -> bool:
                if self.num_timesteps % self.eval_freq == 0:
                    # Train evaluation (with noise)
                    tp, tr, ttrades = [], [], []
                    for i in range(self.n_eval_episodes):
                        obs, _ = self.train_env.reset(random_start=True); done=False; p_t=0.0; r_t=0.0; trade_t=0; steps=0
                        buy_t = sell_t = hold_t = 0
                        while not done:
                            act, _ = self.model.predict(obs, deterministic=True)
                            obs, rew, done, _, info = self.train_env.step(act)
                            real_action = info.get('real_action', act)
                            if real_action == 0:
                                hold_t += 1
                            elif real_action == 1:
                                buy_t += 1; trade_t += 1
                            elif real_action == 2:
                                # Count only successful sells
                                if self.train_env.inventory:
                                    sell_t += 1; trade_t += 1
                                else:
                                    hold_t += 1
                            if real_action == 2:
                                p_t += rew
                            r_t += rew; steps += 1
                        # liquidate remaining positions at end of episode (handle price,qty)
                        if getattr(self.train_env, 'inventory', None):
                            final_price = float(self.train_env.prices[self.train_env.current_step])
                            for bought_price, qty in self.train_env.inventory:
                                profit = (final_price - bought_price) * qty
                                cost = self.train_env.commission * (final_price * qty + bought_price * qty)
                                p_t += profit - cost
                            self.train_env.inventory.clear()
                        tp.append(p_t); tr.append(r_t/steps if steps else 0.0); ttrades.append(trade_t)
                        print(f"[Eval] train ep {i}: profit {p_t:.6f}, buys {buy_t}, sells {sell_t}, holds {hold_t}")
                        logging.info(f"[Eval] train ep {i}: profit {p_t:.6f}, buys {buy_t}, sells {sell_t}, holds {hold_t}")
                    # Validation evaluation (with noise)
                    vp, vr, vtrades = [], [], []
                    for i in range(self.n_eval_episodes):
                        obs, _ = self.eval_env.reset(random_start=True); done=False; p_v=0.0; r_v=0.0; trade_v=0; steps=0
                        buy_v = sell_v = hold_v = 0
                        while not done:
                            act, _ = self.model.predict(obs, deterministic=True)
                            obs, rew, done, _, info = self.eval_env.step(act)
                            real_action = info.get('real_action', act)
                            if real_action == 0:
                                hold_v += 1
                            elif real_action == 1:
                                buy_v += 1; trade_v += 1
                            elif real_action == 2:
                                if self.eval_env.inventory:
                                    sell_v += 1; trade_v += 1
                                else:
                                    hold_v += 1
                            if real_action == 2:
                                p_v += rew
                            r_v += rew; steps += 1
                        # liquidate remaining positions at end of episode (handle price,qty)
                        if getattr(self.eval_env, 'inventory', None):
                            final_price = float(self.eval_env.prices[self.eval_env.current_step])
                            for bought_price, qty in self.eval_env.inventory:
                                profit = (final_price - bought_price) * qty
                                cost = self.eval_env.commission * (final_price * qty + bought_price * qty)
                                p_v += profit - cost
                            self.eval_env.inventory.clear()
                        vp.append(p_v); vr.append(r_v/steps if steps else 0.0); vtrades.append(trade_v)
                        print(f"[Eval] val   ep {i}: profit {p_v:.6f}, buys {buy_v}, sells {sell_v}, holds {hold_v}")
                        logging.info(f"[Eval] val   ep {i}: profit {p_v:.6f}, buys {buy_v}, sells {sell_v}, holds {hold_v}")
                    # Print individual episode profits for clarity
                    best_train, worst_train = max(tp), min(tp)
                    best_val, worst_val = max(vp), min(vp)
                    print(f"[Eval] Profits Train eps: {', '.join(f'{p:.6f}' for p in tp)}")
                    print(f"[Eval] Best/Worst Train profits: {best_train:.6f}/{worst_train:.6f}")
                    print(f"[Eval] Profits Val eps  : {', '.join(f'{p:.6f}' for p in vp)}")
                    print(f"[Eval] Best/Worst Val   profits: {best_val:.6f}/{worst_val:.6f}")
                    # Also log for file
                    for i, p in enumerate(tp): logging.info(f"[Eval] train ep {i}:profit {p:.6f}")
                    for i, p in enumerate(vp): logging.info(f"[Eval] val   ep {i}:profit {p:.6f}")
                    mpt, mrt, mtt = np.mean(tp), np.mean(tr), np.mean(ttrades)
                    mpv, mrv, mv = np.mean(vp), np.mean(vr), np.mean(vtrades)
                    num_val_trades = mv
                    # === Новые метрики: Sharpe, max drawdown, CVaR, Omega, композитный score ===
                    from trading_bot.utils import omega
                    if len(vp) > 1:
                        std_vp = np.std(vp, ddof=1)
                        sharpe = mpv / std_vp if std_vp > 0 else 0.0
                    else:
                        sharpe = 0.0
                    train_max_dd = max_drawdown(tp)
                    val_max_dd = max_drawdown(vp)
                    train_cvar = cvar(tp)
                    val_cvar = cvar(vp)
                    train_omega = omega(tp)
                    val_omega = omega(vp)
                    alpha, beta, gamma, delta = 0.5, 0.2, 0.2, 0.2
                    val_score = mpv - alpha*std_vp - beta*abs(worst_val) - gamma*val_max_dd - delta*val_cvar
                    train_score = mpt - alpha*np.std(tp, ddof=1) - beta*abs(worst_train) - gamma*train_max_dd - delta*train_cvar
                    # Логируем результаты
                    logging.info(f"[Eval] Step {self.num_timesteps}: TrainProfit {mpt:.6f}, ValProfit {mpv:.6f}, Sharpe {sharpe:.6f}, AvgReward {mrv:.6f}, TradesTrain {mtt}, TradesVal {mv}")
                    logging.info(f"[Eval] MaxDD train {train_max_dd:.6f}, val {val_max_dd:.6f}; CVaR train {train_cvar:.6f}, val {val_cvar:.6f}; Omega train {train_omega:.6f}, val {val_omega:.6f}")
                    # early stopping по профиту
                    if mpv > self.best_val_profit:
                        self.best_val_profit = mpv
                        self.no_improve = 0
                        self.model.save(self.save_path)
                        logging.info(f"[Eval] Model improved and saved to {self.save_path}")
                    else:
                        self.no_improve += 1
                        logging.info(f"[Eval] No improvement for {self.no_improve} evals")
                    if self.no_improve >= self.patience:
                        logging.info(f"[Eval] Early stopping triggered at step {self.num_timesteps}")
                        return False
                    return True
                env = TradingEnv(raw_train_prices, raw_train_volumes, window_size, dual_phase=False)
                # enforce same normalization bounds for train and val envs
                train_eval_env = TradingEnv(
                    raw_train_prices, raw_train_volumes, window_size,
                    commission=0.0, max_inventory=1000, carry_cost=0.0,
                    min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
                    risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
                )
                eval_env = TradingEnv(
                    raw_val_prices, raw_val_volumes, window_size,
                    commission=0.0, max_inventory=1000, carry_cost=0.0,
                    min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
                    risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
                )
                # TD3: policy, noise, model, регуляризация, callbacks
                n_actions = env.action_space.shape[0]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
                policy_kwargs = dict(net_arch=[256, 256, 128])
                model = TD3(
                    'MlpPolicy',
                    env,
                    policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    batch_size=256,
                    buffer_size=1_000_000,
                    action_noise=action_noise
                )
                for optimizer in (model.actor.optimizer, model.critic.optimizer):
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] = 1e-5
                save_path = f'{td3_save_name}_{os.path.splitext(stock)[0]}'
                tqdm_cb = TqdmCallback(td3_timesteps)
                eval_cb = EvalCallbackTD3(
                    train_eval_env,
                    eval_env,
                    eval_freq=5000,
                    n_eval_episodes=10,
                    patience=15,
                    save_path=save_path
                )
                model.learn(total_timesteps=td3_timesteps, callback=[tqdm_cb, eval_cb])
                # --- Финальный инференс на трейне ---
                print("=== Final Training Inference TD3 ===")
                logging.info("=== Final Training Inference TD3 ===")
                env_train = TradingEnv(
                    raw_train_prices, raw_train_volumes, window_size,
                    commission=0.0, max_inventory=1000, carry_cost=0.0,
                    min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
                    risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
                )
                obs, _ = env_train.reset()
                done = False
                total_profit_train = 0.0
                trades_train = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    if action != 0:
                        trades_train += 1
                    obs, reward, done, _, _ = env_train.step(action)
                    total_profit_train += reward
                if getattr(env_train, 'inventory', None):
                    final_price = float(env_train.prices[env_train.current_step])
                    for bought_price, qty in env_train.inventory:
                        total_profit_train += (final_price - bought_price) * qty
                    env_train.inventory.clear()
                print(f'Training Total Profit: {float(total_profit_train):.4f}, Trades: {trades_train}')
                logging.info(f'Training Total Profit: {float(total_profit_train):.4f}, Trades: {trades_train}')
                # --- Финальный инференс по валидации после обучения ---
                env_val = TradingEnv(
                    raw_val_prices, raw_val_volumes, window_size,
                    commission=0.0, max_inventory=1000, carry_cost=0.0,
                    min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
                    risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
                )
                obs, _ = env_val.reset()
                done = False
                total_profit = 0.0
                trades_val = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    if action != 0:
                        trades_val += 1
                    obs, reward, done, _, _ = env_val.step(action)
                    total_profit += reward
                if getattr(env_val, 'inventory', None):
                    final_price = float(env_val.prices[env_val.current_step])
                    for bought_price, qty in env_val.inventory:
                        total_profit += (final_price - bought_price) * qty
                    env_val.inventory.clear()
                print(f'Validation Total Profit: {float(total_profit):.4f}, Trades: {trades_val}')
                logging.info(f'Validation Total Profit: {float(total_profit):.4f}, Trades: {trades_val}')
        #     raw_val_prices, raw_val_volumes, window_size,
        #     commission=0.001, max_inventory=8, carry_cost=0.0001,
        #     min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
        #     risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True
        # )
        # risk_cb = EvalCallbackTD3(
        #     risk_train_env, risk_eval_env, eval_freq=5000,
        #     n_eval_episodes=10, patience=15, save_path=save_path+"_risk"
        # )
        # model.learn(total_timesteps=td3_timesteps//2, callback=[tqdm_cb, risk_cb])

    pbar = tqdm(range(1, ep_count+1), desc="Finetune Epoch")
    init_thr, final_thr = -0.01, 0.01  # dynamic threshold schedule
    for epoch in pbar:
        # update threshold: start aggressive, end conservative
        agent.buy_threshold = init_thr + (final_thr - init_thr) * (epoch - 1) / (ep_count - 1)
        result = train_model(agent, epoch, train_data, ep_count=ep_count, batch_size=batch_size, window_size=window_size)
        # Оценим на трейне векторизованно (batch predict)
        states = np.vstack([get_state(train_data, i, window_size)[0] for i in range(len(train_data)-1)])
        qvals = agent.model.predict(states, verbose=0)
        actions = []
        inventory = []
        for q, price in zip(qvals, raw_train_prices[:-1]):
            if q[1] - q[0] > agent.buy_threshold:
                actions.append(1)
                inventory.append(price)
            elif q[2] - q[0] > agent.buy_threshold and inventory:
                actions.append(2)
                inventory.pop(0)
            else:
                actions.append(0)
        profit = 0.0
        position = []
        deltas = []
        valid_trades = 0
        for act, price in zip(actions, raw_train_prices[:-1]):
            if act == 1:
                position.append(price)
                valid_trades += 1
            elif act == 2 and position:
                delta = price - position.pop(0)
                profit += delta
                deltas.append(delta)
                valid_trades += 1
        for price in position:
            delta = raw_train_prices[-1] - price
            profit += delta
            deltas.append(delta)
        trades = valid_trades
        # Annualized Sharpe ratio (per trade): mean / std * sqrt(N)
        N = len(deltas)
        sharpe = (np.mean(deltas) / (np.std(deltas) + 1e-8)) * np.sqrt(N) if N > 1 else 0.0
        train_omega = omega(deltas) if len(deltas) > 1 else 0.0
        train_cvar = cvar(deltas) if len(deltas) > 1 else 0.0
        train_max_dd = max_drawdown(deltas) if len(deltas) > 1 else 0.0
        print(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f} omega={train_omega:.2f} maxDD={train_max_dd:.2f} cvar={train_cvar:.2f}")
        logging.info(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f} omega={train_omega:.2f} maxDD={train_max_dd:.2f} cvar={train_cvar:.2f}")
        with open("train_finetune.log", "a") as f:
            f.write(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f} omega={train_omega:.2f} maxDD={train_max_dd:.2f} cvar={train_cvar:.2f}\n")
        # Evaluate on val set
        val_profit, val_deltas = evaluate_model(agent, val_data, window_size, debug, min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices), return_deltas=True)
        # Annualized Sharpe for val
        N_val = len(val_deltas)
        sharpe_val = (np.mean(val_deltas) / (np.std(val_deltas) + 1e-8)) * np.sqrt(N_val) if N_val > 1 else 0.0
        val_omega = omega(val_deltas) if len(val_deltas) > 1 else 0.0
        val_cvar = cvar(val_deltas) if len(val_deltas) > 1 else 0.0
        val_max_dd = max_drawdown(val_deltas) if len(val_deltas) > 1 else 0.0
        logging.info(f"Epoch {epoch}/{ep_count}: val_profit={val_profit:.2f} val_sharpe={sharpe_val:.2f} omega={val_omega:.2f} maxDD={val_max_dd:.2f} cvar={val_cvar:.2f}")
        print(f"Epoch {epoch}/{ep_count}: val_profit={val_profit:.2f} val_sharpe={sharpe_val:.2f} omega={val_omega:.2f} maxDD={val_max_dd:.2f} cvar={val_cvar:.2f}")
        with open("train_finetune.log", "a") as f:
            f.write(f"Epoch {epoch}/{ep_count}: val_profit={val_profit:.2f} val_sharpe={sharpe_val:.2f} omega={val_omega:.2f} maxDD={val_max_dd:.2f} cvar={val_cvar:.2f}\n")
        pbar.set_postfix(train_profit=f"{profit:.2f}", val_profit=f"{val_profit:.2f}", sharpe=f"{sharpe:.2f}", val_sharpe=f"{sharpe_val:.2f}", omega=f"{train_omega:.2f}/{val_omega:.2f}", maxDD=f"{train_max_dd:.2f}/{val_max_dd:.2f}", cvar=f"{train_cvar:.2f}/{val_cvar:.2f}")
        if best_val_profit is None or val_profit > best_val_profit:
            best_val_profit = val_profit
            best_val_epoch = epoch
            agent.model.save_weights(f"models/best_{strat}_{model_name}_{window_size}.weights.h5")
            save_window_size(f"models/best_{strat}_{model_name}_{window_size}.weights.h5", window_size)
        else:
            no_improve += 1
        if earlystop_patience and no_improve >= earlystop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")
    logging.info(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")

if __name__ == '__main__':
    print("Запуск блока __main__")
    try:
        # Исправляем обработку аргументов docopt
        if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
            # Если первый аргумент - CSV файл, то используем его напрямую
            stock = sys.argv[1]
            print(f"Путь к файлу данных (прямой): {stock}")
        else:
            # Иначе используем docopt
            args = docopt(__doc__)
            print(f"Аргументы успешно парсинг docopt: {args}")
            stock = args['<stock>']
            print(f"Путь к файлу данных: {stock}")
        
        # Проверяем существование файла с относительным и абсолютным путем
        if not os.path.isfile(stock):
            # Пробуем искать в подкаталоге data
            data_path = os.path.join('data', stock)
            if os.path.isfile(data_path):
                stock = data_path
                print(f"Файл найден в подкаталоге data: {stock}")
            else:
                print(f"Error: Файл {stock} не найден ни в текущем каталоге, ни в подкаталоге data")
                print(__doc__)
                exit(1)
        if not stock.lower().endswith('.csv'):
            print(f"Error: Файл должен быть в формате CSV. Получено: {stock}")
            print(__doc__)
            exit(1)
        print(f"Используем файл данных: {stock}")
        
        # Парсим остальные аргументы
        if 'args' in locals():
            # Если использовали docopt
            window_size = int(args["--window-size"] or 20)
            batch_size = int(args["--batch-size"] or 32)
            ep_count = int(args["--episode-count"] or 50)
            strategy = args["--strategy"] or "td3"
            model_name = args["--model-name"] or "model_debug"
            pretrained = args["--pretrained"]
            debug = args["--debug"]
            model_type = args["--model-type"] or "dense"
            target_update = int(args["--target-update"] or 100)
            td3_timesteps = int(args["--td3-timesteps"] or 100000)
            td3_noise_sigma = float(args["--td3-noise-sigma"] or 1.0)
            td3_save_name = args["--td3-save-name"] or "td3_model"
            force_cpu = args["--force-cpu"]
        else:
            # Используем значения по умолчанию и прямой парсинг аргументов
            window_size = 20
            batch_size = 32
            ep_count = 50
            strategy = "td3"
            model_name = "model_debug"
            pretrained = False
            debug = "--debug" in sys.argv
            model_type = "dense"
            target_update = 100
            td3_timesteps = 100000
            force_cpu = "--force-cpu" in sys.argv
            
            # Парсим числовые аргументы
            for i, arg in enumerate(sys.argv):
                if arg == "--window-size" and i+1 < len(sys.argv):
                    window_size = int(sys.argv[i+1])
                elif arg == "--batch-size" and i+1 < len(sys.argv):
                    batch_size = int(sys.argv[i+1])
                elif arg == "--episode-count" and i+1 < len(sys.argv):
                    ep_count = int(sys.argv[i+1])
                elif arg == "--strategy" and i+1 < len(sys.argv):
                    strategy = sys.argv[i+1]
                elif arg == "--model-name" and i+1 < len(sys.argv):
                    model_name = sys.argv[i+1]
                elif arg == "--model-type" and i+1 < len(sys.argv):
                    model_type = sys.argv[i+1]
                elif arg == "--target-update" and i+1 < len(sys.argv):
                    target_update = int(sys.argv[i+1])
                elif arg == "--td3-timesteps" and i+1 < len(sys.argv):
                    td3_timesteps = int(sys.argv[i+1])
                elif arg == "--td3-noise-sigma" and i+1 < len(sys.argv):
                    td3_noise_sigma = float(sys.argv[i+1])
                elif arg == "--td3-save-name" and i+1 < len(sys.argv):
                    td3_save_name = sys.argv[i+1]
                elif arg == "--pretrained":
                    pretrained = True
            
            # Если не задано имя модели TD3, используем значение по умолчанию
            if "td3_save_name" not in locals():
                td3_save_name = "td3_model"
            if "td3_noise_sigma" not in locals():
                td3_noise_sigma = 1.0
        
        print("Все аргументы успешно парсинг")
        
        if force_cpu:
            print("Режим: принудительное использование CPU")
        print(f"ARGS:\n  stock={stock}\n  window_size={window_size}\n  batch_size={batch_size}\n  ep_count={ep_count}\n  strategy={strategy}\n  model_name={model_name}\n  pretrained={pretrained}\n  debug={debug}\n  model_type={model_type}\n  target_update={target_update}\n  td3_timesteps={td3_timesteps}\n  td3_noise_sigma={td3_noise_sigma}\n  td3_save_name={td3_save_name}")
    except Exception as e:
        print(f"Ошибка при парсинге аргументов: {e}")
        traceback.print_exc()
        sys.exit(1)
    # Применяем настройки для модели LSTM
    if model_type == 'lstm':
        ep_count = 500
        earlystop_patience = 50
    else:
        earlystop_patience = None

    # Для TD3 фиксируем window_size=47
    if strategy.lower() == 'td3':
        window_size = 47  # Жёстко фиксируем window_size=47 для TD3
        
    # Определяем имя модели для сохранения
    model_save_name = td3_save_name if strategy.lower() == 'td3' else model_name
    
    # Запускаем основную функцию обучения
    try:
        print("Запуск main() с параметрами:")
        print(f"  stock={stock}")
        print(f"  window_size={window_size}")
        print(f"  batch_size={batch_size}")
        print(f"  ep_count={ep_count}")
        print(f"  strategy={strategy}")
        print(f"  model_name={model_name}")
        print(f"  pretrained={pretrained}")
        print(f"  debug={debug}")
        print(f"  target_update={target_update}")
        print(f"  td3_timesteps={td3_timesteps}")
        print(f"  td3_noise_sigma={td3_noise_sigma}")
        print(f"  td3_save_name={td3_save_name}")
        
        # Вызываем функцию main с параметрами
        main(stock, window_size, batch_size, ep_count, strategy, model_name, pretrained, debug, target_update, td3_timesteps, td3_noise_sigma, td3_save_name)
        
        # Сохраняем window_size в модель после обучения
        if model_save_name:
            try:
                from stable_baselines3 import TD3
                if os.path.exists(model_save_name):
                    model = TD3.load(model_save_name)
                    model.save(model_save_name, user_data={"window_size": window_size})
                    print(f"Успешно сохранили window_size={window_size} в модель {model_save_name}")
            except Exception as e:
                print(f"Ошибка при сохранении window_size в модель: {e}")
                print(f"[WARNING] Не удалось сохранить window_size в user_data: {e}")
        
        print("Скрипт успешно завершен.")
    except Exception as e:
        print(f"[ERROR] Ошибка в основном блоке: {e}")
        traceback.print_exc()
