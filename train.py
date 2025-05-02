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

"""

import logging
import coloredlogs
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import io
import re
import numpy as np
from datetime import datetime

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    WINDOW_SIZE,
    minmax_normalize,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device,
    zscore_normalize
)

# Configure multi-core usage
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

def main(stock, window_size=WINDOW_SIZE, batch_size=32, ep_count=50,
         strategy="t-dqn", model_name=None, pretrained=False,
         debug=False, model_type='dense', target_update=100,
         td3_timesteps=100000, td3_noise_sigma=1.0, td3_save_name='td3_model'):
    """ Finetune the stock trading bot on a large interval (2019-01-01 — 2024-06-30).
    Logs each epoch to train_finetune.log, uses tqdm for progress.
    """
    import pandas as pd
    import os
    import logging
    from trading_bot.ops import get_state
    from tqdm import tqdm

    # Настраиваем логирование
    if debug:
        log_file = "train_debug.log"
    else:
        log_file = "train_finetune.log"
    
    # Очищаем лог-файл перед началом
    with open(log_file, 'w') as f:
        f.write("")
    
    print(f"Лог обучения будет писаться в {log_file}")

    # === TD3 TRAINING BLOCK ===
    if strategy == 'td3':
        import numpy as np
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
        from trading_bot.env import TradingEnv
        from trading_bot.visualize_callback import VisualizeCallback
        import pandas as pd
        from tqdm import tqdm
        import time
        import os
        
        # Создаем директории для моделей и графиков
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print("TD3 Training (manual): подготовка данных...")
        
        # Загружаем данные
        data = pd.read_csv(stock)
        data = data.sort_values('Date')
        
        # Разделяем на тренировочные и валидационные данные
        train_data = data.iloc[:-20]
        val_data = data.iloc[-20:]
        
        print(f"Тренировочные данные: {len(train_data)} дней")
        print(f"Валидационные данные: {len(val_data)} дней")
        
        # Получаем цены и объемы для обучения
        train_prices = train_data['Close'].values
        train_volumes = train_data['Volume'].values
        val_prices = val_data['Close'].values
        val_volumes = val_data['Volume'].values
        
        # Создаем среды
        train_env = TradingEnv(train_prices, train_volumes, window_size)
        val_env = TradingEnv(val_prices, val_volumes, window_size)
        
        # Сохраняем оригинальные цены для расчета прибыли
        orig_train_prices = train_prices
        orig_val_prices = val_prices
        
        # Создаем callback для прогресс-бара и логирования
        class TD3EvalCallback(BaseCallback):
            def __init__(self, train_env, val_env, eval_freq=1000, n_eval_episodes=10, log_file=None):
                super().__init__()
                self.train_env = train_env
                self.val_env = val_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes
                self.log_file = log_file
                self.best_mean_reward = -float('inf')
                self.pbar = None
                self.eval_count = 0
                
                # Сохраняем оригинальные цены
                self.train_prices = orig_train_prices
                self.val_prices = orig_val_prices
                
                # Для отслеживания шагов оценки
                self.next_eval_step = eval_freq  # Следующий шаг для оценки
                self.is_evaluating = False  # Флаг, указывающий, что идет оценка
                
                # Для хранения результатов тренировки
                self.train_profit = 0
                self.train_sharpe = 0
                
            def _on_training_start(self):
                # Инициализируем прогресс-бар без обновления
                self.pbar = tqdm(total=td3_timesteps, desc="TD3 Training", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                self._log(f"Начало обучения TD3. Всего шагов: {td3_timesteps}")
                
            def _on_step(self):
                # Обновляем прогресс-бар только если не идет оценка
                if not self.is_evaluating:
                    self.pbar.update(1)
                
                # Проверяем, нужно ли начать оценку
                if self.n_calls == self.next_eval_step and not self.is_evaluating:
                    self.is_evaluating = True  # Устанавливаем флаг оценки
                    self.eval_count += 1
                    
                    # Закрываем текущий прогресс-бар перед оценкой
                    self.pbar.close()
                    
                    self._log(f"=== Оценка #{self.eval_count} (шаг {self.n_calls}) ===")
                    
                    # Оценка на тренировочных данных (только 2 эпизода для ускорения)
                    train_rewards, train_buys, train_sells, train_holds, train_profits, train_sharpe = self._evaluate_env(
                        self.train_env, self.train_prices, n_episodes=2, is_train=True)
                    
                    # Оценка на валидационных данных (только 2 эпизода для ускорения)
                    val_rewards, val_buys, val_sells, val_holds, val_profits, val_sharpe = self._evaluate_env(
                        self.val_env, self.val_prices, n_episodes=2, is_train=False)
                    
                    # Сохранение лучшей модели по Sharpe Ratio
                    if val_sharpe > self.best_mean_reward:
                        self.best_mean_reward = val_sharpe
                        self.model.save(f"models/{td3_save_name}_best")
                        self._log(f"Новая лучшая модель сохранена с val_sharpe={val_sharpe:.2f}")
                    
                    # Устанавливаем следующий шаг для оценки
                    self.next_eval_step += self.eval_freq
                    
                    # Создаем новый прогресс-бар с обновленными метриками
                    self.pbar = tqdm(
                        total=td3_timesteps, 
                        initial=self.n_calls,
                        desc="TD3 Training",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        postfix={
                            'train_profit': f"{train_profits:.2f}", 
                            'val_profit': f"{val_profits:.2f}", 
                            'train_sharpe': f"{train_sharpe:.2f}", 
                            'val_sharpe': f"{val_sharpe:.2f}"
                        }
                    )
                    
                    # Сбрасываем флаг оценки
                    self.is_evaluating = False
                
                return True
                
            def _on_training_end(self):
                self.pbar.close()
                self._log("Обучение завершено")
                
            def _evaluate_env(self, env, orig_prices, n_episodes=None, is_train=True):
                """Оценка модели на среде с подсчетом метрик"""
                # Если n_episodes не указано, используем значение из self.n_eval_episodes
                if n_episodes is None:
                    n_episodes = self.n_eval_episodes
                
                total_rewards = 0
                buys, sells, holds = 0, 0, 0
                total_profit = 0
                all_returns = []  # Для расчета Sharpe Ratio
                episode_profits = []  # Для отслеживания прибыли по эпизодам
                
                # Сбрасываем инвентарь среды перед оценкой
                env.inventory = []
                
                for ep in range(n_episodes):
                    obs, _ = env.reset()
                    done = False
                    episode_reward = 0
                    positions = []  # Для отслеживания открытых позиций
                    episode_returns = []  # Доходность для текущего эпизода
                    episode_profit = 0  # Прибыль для текущего эпизода
                    ep_buys, ep_sells, ep_holds = 0, 0, 0  # Счетчики действий для эпизода
                    
                    # Для логирования сделок
                    buy_prices = []
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done, _, info = env.step(action)
                        episode_reward += reward
                        
                        # Считаем действия
                        real_action = info.get('real_action', 0)
                        if real_action == 1:  # Покупка
                            buys += 1
                            ep_buys += 1
                            # Запоминаем цену покупки (используем оригинальные цены)
                            price_idx = min(env.current_step, len(orig_prices)-1)
                            buy_price = orig_prices[price_idx]
                            positions.append(buy_price)
                            buy_prices.append(buy_price)
                            
                            # Логируем в точном формате из примера
                            with open(self.log_file, 'a') as f:
                                f.write(f"DEBUG Buy at: ${buy_price:.2f}\n")
                        elif real_action == 2:  # Продажа
                            sells += 1
                            ep_sells += 1
                            # Рассчитываем прибыль, если есть открытая позиция
                            if positions:
                                buy_price = positions.pop(0)
                                price_idx = min(env.current_step, len(orig_prices)-1)
                                sell_price = orig_prices[price_idx]
                                profit = sell_price - buy_price
                                total_profit += profit
                                episode_profit += profit
                                
                                # Логируем продажу в точном формате из примера
                                with open(self.log_file, 'a') as f:
                                    f.write(f"DEBUG Sell at: ${sell_price:.2f} | Position: {'+' if profit >= 0 else ''}{profit:.2f}\n")
                                
                                # Рассчитываем доходность для Sharpe Ratio
                                if buy_price > 0:  # Защита от деления на ноль
                                    returns_pct = (sell_price - buy_price) / buy_price
                                    episode_returns.append(returns_pct)
                        else:  # Удержание
                            holds += 1
                            ep_holds += 1
                    
                    # Закрываем оставшиеся позиции по последней цене
                    if positions and len(orig_prices) > 0:
                        last_price = orig_prices[-1]
                        for buy_price in positions:
                            profit = last_price - buy_price
                            total_profit += profit
                            episode_profit += profit
                            
                            # Логируем закрытие позиции в точном формате из примера
                            with open(self.log_file, 'a') as f:
                                f.write(f"DEBUG Close at: ${last_price:.2f} | Position: {'+' if profit >= 0 else ''}{profit:.2f}\n")
                            
                            # Рассчитываем доходность для Sharpe Ratio
                            if buy_price > 0:  # Защита от деления на ноль
                                returns_pct = (last_price - buy_price) / buy_price
                                episode_returns.append(returns_pct)
                    
                    total_rewards += episode_reward
                    
                    # Добавляем доходности эпизода в общий список
                    if episode_returns:
                        all_returns.extend(episode_returns)
                    
                    # Сохраняем прибыль эпизода
                    episode_profits.append(episode_profit)
                    
                    # Логируем результаты эпизода
                    env_type = "train" if is_train else "val  "
                    self._log(f"[Eval] {env_type} ep {ep}:profit {episode_profit:.6f}")
                
                # Средние значения
                avg_reward = total_rewards / max(1, n_episodes)
                avg_profit = total_profit / max(1, n_episodes)
                
                # Расчет Sharpe Ratio
                sharpe_ratio = 0.0
                if len(all_returns) > 1:
                    mean_return = np.mean(all_returns)
                    std_return = np.std(all_returns)
                    
                    # Годовой Sharpe с защитой от деления на ноль
                    if std_return > 1e-8:
                        sharpe_ratio = mean_return / std_return * np.sqrt(252)
                
                # Ограничиваем Sharpe разумными пределами
                sharpe_ratio = np.clip(sharpe_ratio, -5.0, 5.0)
                
                # Сохраняем результаты тренировки для итогового лога
                if is_train:
                    self.train_profit = total_profit
                    self.train_sharpe = sharpe_ratio
                else:
                    # Формат точно как в примере - записываем только в конце валидации
                    with open(self.log_file, 'a') as f:
                        f.write(f"INFO Episode {self.eval_count}/50 - Train Position: +${self.train_profit:.2f}  Val Position: +${total_profit:.2f}  Train Loss: {self.train_sharpe:.4f})\n")
                
                return avg_reward, buys, sells, holds, total_profit, sharpe_ratio
                
            def _log(self, message):
                """Логирование сообщения в файл"""
                with open(self.log_file, 'a') as f:
                    f.write(message + '\n')
        
        # Настраиваем модель TD3
        n_actions = train_env.action_space.shape[0] if hasattr(train_env.action_space, 'shape') else train_env.action_space.n
        
        print("TD3 Training: запуск обучения на", td3_timesteps, "шагов...")
        
        # Создаем модель TD3
        model = TD3(
            "MlpPolicy",
            train_env,
            verbose=0,
            buffer_size=100000,
            learning_rate=3e-4,
            batch_size=256,
            train_freq=(1, "episode"),
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=td3_noise_sigma * np.ones(n_actions)
            ),
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[64, 64],
                    qf=[64, 64]
                )
            )
        )
        
        # Создаем callback для визуализации
        vis_callback = VisualizeCallback(
            train_env=train_env,
            val_env=val_env,
            model=model,
            total_timesteps=td3_timesteps,
            max_plots=10
        )
        
        # Создаем callback для оценки
        eval_callback = TD3EvalCallback(
            train_env=train_env,
            val_env=val_env,
            eval_freq=1000,
            n_eval_episodes=2,
            log_file=log_file
        )
        
        # Обучаем модель
        model.learn(
            total_timesteps=td3_timesteps,
            callback=[eval_callback, vis_callback]
        )
        
        # Сохраняем финальную модель
        model.save(f"models/{td3_save_name}_final")
        
        # Оцениваем модель на валидационных данных
        val_env.reset()
        obs = val_env._get_observation()
        done = False
        total_reward = 0
        steps = 0
        buys, sells, holds = 0, 0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = val_env.step(action)
            total_reward += reward
            steps += 1
            
            real_action = info.get('real_action', 0)
            if real_action == 1:  # Покупка
                buys += 1
                price_idx = min(val_env.current_step, len(orig_val_prices)-1)
                buy_price = orig_val_prices[price_idx]
                with open(log_file, 'a') as f:
                    f.write(f"DEBUG Buy at: ${buy_price:.2f}\n")
            elif real_action == 2:  # Продажа
                sells += 1
                price_idx = min(val_env.current_step, len(orig_val_prices)-1)
                sell_price = orig_val_prices[price_idx]
                with open(log_file, 'a') as f:
                    f.write(f"DEBUG Sell at: ${sell_price:.2f}\n")
            else:  # Удержание
                holds += 1
        
        print(f"Валидация: reward={total_reward:.2f}, buys={buys}, sells={sells}, holds={holds}")
        
        # Запись в лог-файл
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ФИНАЛЬНАЯ ОЦЕНКА: reward={total_reward:.2f}, buys={buys}, sells={sells}, holds={holds}\n")
        
        exit(0)
    # === END TD3 TRAINING BLOCK ===

    # load and clean CSV for train
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
    # train: всё из stock (например, GOOG_2010-2024-06.csv)
    train_df = df.copy()
    train_prices = train_df['Adj Close'].values
    train_vols = train_df['Volume'].values
    # out-of-sample validation: отдельный файл
    val_df = pd.read_csv('data/GOOG_2024-07_2025-04.csv')
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    val_df = val_df.sort_values('Date').reset_index(drop=True)
    val_prices = val_df['Adj Close'].values
    val_vols = val_df['Volume'].values
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days (OOS)")
    raw_train_prices = train_df["Adj Close"].values
    raw_train_volumes = train_df["Volume"].values
    # normalize price and volume and combine
    price_norm = list(minmax_normalize(raw_train_prices))
    vol_norm = list(minmax_normalize(raw_train_volumes))
    train_data = list(zip(price_norm, vol_norm))
    # prepare validation data and normalize separately
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
        import numpy as np
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

        train_env = TradingEnv(train_prices, train_vols, window_size)
        val_env = TradingEnv(val_prices, val_vols, window_size)
        #train_env = Monitor(train_env, monitor_dir)  # Отключено для совместимости с визуализацией
        n_actions = train_env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
        # Initialize TD3 with TensorBoard logging
        model = TD3('MlpPolicy', train_env, action_noise=action_noise, verbose=1,
                    tensorboard_log=tb_dir)
        # attach visualization callback
        visual_cb = VisualizeCallback(train_env, val_env, model, total_timesteps=td3_timesteps)
        # Явно выводим прогресс через tqdm
        from tqdm import tqdm
        pbar = tqdm(total=td3_timesteps, desc='TD3 Training (manual)')
        callback_steps = td3_timesteps // 100 if td3_timesteps > 100 else 1
        class TqdmProgressCallback:
            def __init__(self, steps_per_update):
                self.steps_per_update = steps_per_update
                self.counter = 0
            def __call__(self, _locals, _globals):
                self.counter += 1
                if self.counter % self.steps_per_update == 0 or self.counter == td3_timesteps:
                    pbar.update(self.steps_per_update)
                return True
            def __getattr__(self, name):
                # Для совместимости с Stable-Baselines3 callback API
                return lambda *a, **kw: None
        tqdm_callback = TqdmProgressCallback(callback_steps)
        model.learn(total_timesteps=td3_timesteps, callback=[visual_cb, tqdm_callback])
        pbar.close()
        model.save(f"{td3_save_name}")
        #df = pd.read_csv(os.path.join(monitor_dir, 'monitor.csv'), comment='#')
        #plt.figure(figsize=(8,4))
        #plt.plot(df['l'], df['r'], label='Episode Reward')
        #plt.xlabel('Episode')
        #plt.ylabel('Reward')
        #plt.title('Training Reward Progression')
        #plt.legend()
        #plt.tight_layout()
        #plt.savefig(os.path.join(plots_dir, 'training_reward.png'))
        #plt.close()
        return
    # если выбран TD3, запускаем Stable-Baselines3 TD3
    if strategy.lower() == 'td3':
        import os
        import numpy as np
        import pandas as pd
        from trading_bot.env import TradingEnv
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
        from tqdm import tqdm
        import torch
        import logging
        print("[DEBUG] TD3 section: imports инициализированы")
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
                            if real_action == 1:  # Покупка
                                buy_t += 1
                                trade_t += 1
                            elif real_action == 2:  # Продажа
                                sell_t += 1
                                if self.train_env.inventory:
                                    trade_t += 1
                                else:
                                    hold_t += 1
                            else:  # Удержание
                                hold_t += 1
                            if real_action == 2:
                                p_t += rew
                            r_t += rew; steps += 1
                        # liquidate remaining positions at end of episode (handle price,qty)
                        if getattr(self.train_env, 'inventory', None):
                            final_price = float(self.train_env.prices[self.train_env.current_step])
                            for bought_price, qty in self.train_env.inventory:
                                profit = (final_price - bought_price) * qty
                                p_t += profit
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
                            if real_action == 1:  # Покупка
                                buy_v += 1
                                trade_v += 1
                            elif real_action == 2:  # Продажа
                                sell_v += 1
                                if self.eval_env.inventory:
                                    trade_v += 1
                                else:
                                    hold_v += 1
                            else:  # Удержание
                                hold_v += 1
                            if real_action == 2:
                                p_v += rew
                            r_v += rew; steps += 1
                        # liquidate remaining positions at end of episode (handle price,qty)
                        if getattr(self.eval_env, 'inventory', None):
                            final_price = float(self.eval_env.prices[self.eval_env.current_step])
                            for bought_price, qty in self.eval_env.inventory:
                                profit = (final_price - bought_price) * qty
                                p_v += profit
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
                    # Compute Sharpe ratio across episodes; avoid huge values when returns constant
                    if len(vp) > 1:
                        std_vp = np.std(vp, ddof=1)
                        sharpe = mpv / std_vp if std_vp > 0 else 0.0
                    else:
                        sharpe = 0.0
                    # Report with high precision
                    msg = (f"[Eval] Step {self.num_timesteps}: "
                           f"TrainProfit {mpt:.6f}, ValProfit {mpv:.6f}, Sharpe {sharpe:.6f}, "
                           f"AvgReward {mrv:.6f}, TradesTrain {mtt:.1f}, TradesVal {mv:.1f}")
                    print(msg); logging.info(msg)
                    # Early stopping based on average validation profit
                    if mpv > self.best_val_profit:
                        self.best_val_profit = mpv
                        self.no_improve = 0
                        self.model.save(self.save_path + '_best')
                    else:
                        self.no_improve += 1
                        if self.no_improve >= self.patience:
                            print(f"Early stopping at step {self.num_timesteps}")
                            return False
                return True

        # создаём среды
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
        # continuous action dim for TD3 (shape of Box)
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
        # Configure TD3 with tuned hyperparameters and deeper MLP architecture
        policy_kwargs = dict(net_arch=[256, 256, 128])  # расширенная сеть
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        env = DummyVecEnv([lambda: env])
        train_eval_env = DummyVecEnv([lambda: train_eval_env])
        eval_env = DummyVecEnv([lambda: eval_env])
        
        policy_kwargs = dict(net_arch=[256, 256, 128])
        model = TD3(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            action_noise=action_noise,
            verbose=1,
            device="cpu"
        )
        # ===== L2 Regularization (Weight Decay) =====
        # Устанавливаем небольшой weight decay для actor и critic оптимизаторов
        for optimizer in (model.actor.optimizer, model.critic.optimizer):
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 1e-5
        # Setup callbacks: progress bar and evaluation with early stopping
        save_path = f'{td3_save_name}_{os.path.splitext(stock)[0]}'
        tqdm_cb = TqdmCallback(td3_timesteps)
        # Increase evaluation stability: more episodes and higher patience for early stopping
        eval_cb = EvalCallbackTD3(
            train_eval_env,
            eval_env,
            eval_freq=5000,
            n_eval_episodes=10,  # ещё больше эпизодов для стабильности
            patience=15,         # дольше ждём улучшения
            save_path=save_path
        )

        print("[DEBUG] Перед model.learn")
        try:
            model.learn(total_timesteps=td3_timesteps, callback=[tqdm_cb, eval_cb])
        except Exception as e:
            print(f"[ERROR] model.learn exception: {e}")
            logging.error(f"[ERROR] model.learn exception: {e}")
            raise
        print("[DEBUG] После model.learn")
        # ===== ЭТАП 1: обучение на чистом профите (без комиссий и рисков)
        # (train_eval_env и eval_env уже без комиссий и risk shaping)
        # ===== ЭТАП 2: фаинтюн с рисками (опционально, пока закомментировано)
        # risk_train_env = TradingEnv(
        #     raw_train_prices, raw_train_volumes, window_size,
        #     commission=0.001, max_inventory=8, carry_cost=0.0001,
        #     min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
        #     risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True
        # )
        # risk_eval_env = TradingEnv(
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

        # Train with callbacks, allow interruption
        # try:
        #     model.learn(total_timesteps=td3_timesteps, callback=[tqdm_cb, eval_cb])
        # except KeyboardInterrupt:
        #     print("Training interrupted by user, proceeding to final inference...")
        # Save final model
        model.save(f'{td3_save_name}_{os.path.splitext(stock)[0]}')
        # Final inference on training set
        print("=== Final Training Inference TD3 ===")
        logging.info("=== Final Training Inference TD3 ===")
        # Final inference environment: disable costs and risk shaping for clear profit calc
        env_train = TradingEnv(
            raw_train_prices, raw_train_volumes, window_size,
            commission=0.0, max_inventory=1000, carry_cost=0.0,
            min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
            risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
        )
        obs, _ = env_train.reset(); done=False; total_profit_train=0.0; trades_train=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if action != 0: trades_train += 1
            obs, reward, done, _, _ = env_train.step(action); total_profit_train += reward
        if getattr(env_train, 'inventory', None):
            final_price = float(env_train.prices[env_train.current_step])
            for bought_price, qty in env_train.inventory:
                total_profit_train += (final_price - bought_price) * qty
            env_train.inventory.clear()
        print(f'Training Total Profit: {float(total_profit_train):.4f}, Trades: {trades_train}')
        logging.info(f'Training Total Profit: {float(total_profit_train):.4f}, Trades: {trades_train}')
        # Final inference on validation set
        print("=== Final Validation Inference TD3 ===")
        logging.info("=== Final Validation Inference TD3 ===")
        # Validation environment: same normalization, no costs/risk shaping
        env_val = TradingEnv(
            raw_val_prices, raw_val_volumes, window_size,
            commission=0.0, max_inventory=1000, carry_cost=0.0,
            min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices),
            risk_lambda=0.0, drawdown_lambda=0.0, dual_phase=False
        )
        obs, _ = env_val.reset(); done=False; total_profit=0.0; trades_val=0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if action != 0: trades_val += 1
            obs, reward, done, _, _ = env_val.step(action); total_profit += reward
        if getattr(env_val, 'inventory', None):
            final_price = float(env_val.prices[env_val.current_step])
            for bought_price, qty in env_val.inventory:
                total_profit += (final_price - bought_price) * qty
            env_val.inventory.clear()
        print(f'Validation Total Profit: {float(total_profit):.4f}, Trades: {trades_val}')
        logging.info(f'Validation Total Profit: {float(total_profit):.4f}, Trades: {trades_val}')
        return
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
        sharpe = (np.mean(deltas) / (np.std(deltas) + 1e-8)) if deltas else 0.0
        print(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f}")
        logging.info(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f}")
        with open("train_finetune.log", "a") as f:
            f.write(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={trades} sharpe={sharpe:.2f}\n")
        # Evaluate on val set
        val_profit, _ = evaluate_model(agent, val_data, window_size, debug, min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices))
        logging.info(f"Epoch {epoch}/{ep_count}: val_profit={val_profit:.2f}")
        pbar.set_postfix(train_profit=f"{profit:.2f}", val_profit=f"{val_profit:.2f}", sharpe=f"{sharpe:.2f}")
        if best_val_profit is None or val_profit > best_val_profit:
            best_val_profit = val_profit
            best_val_epoch = epoch
            agent.model.save_weights(f"models/best_{strat}_{model_name}_{window_size}.weights.h5")
            no_improve = 0
        else:
            no_improve += 1
        if earlystop_patience and no_improve >= earlystop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")
    logging.info(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")


if __name__ == "__main__":
    args = docopt(__doc__)

    stock = args["<stock>"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    strategy = args["--strategy"]
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    model_type = args.get("--model-type") or 'dense'
    target_update = int(args.get("--target-update") or 100)
    td3_timesteps = int(args.get("--td3-timesteps") or 1000000)
    td3_noise_sigma = float(args.get("--td3-noise-sigma") or 1.0)
    td3_save_name = args.get("--td3-save-name") or 'td3_model'

    # LSTM: 500 эпох, earlystop=50
    if model_type == 'lstm':
        ep_count = 500
        earlystop_patience = 50
    else:
        earlystop_patience = None

    main(stock, window_size=window_size, batch_size=batch_size, ep_count=ep_count,
         strategy=strategy, model_name=model_name, pretrained=pretrained, debug=debug, model_type=model_type, target_update=target_update,
         td3_timesteps=td3_timesteps, td3_noise_sigma=td3_noise_sigma, td3_save_name=td3_save_name)
