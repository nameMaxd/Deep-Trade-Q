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
import os
import sys
import traceback
import numpy as np
import pandas as pd
from docopt import docopt
from config import *  # Импортируем все параметры
from trading_bot.TD3_agent import create_TD3_model, train_TD3, train_TD3_with_callbacks  # Импортируем TD3 агента

# Включаем отладочный вывод
# print("=== Запуск скрипта train.py ===")
# print(f"Аргументы: {sys.argv}")
# print(f"Текущая директория: {os.getcwd()}")

# Проверяем наличие файла данных
if len(sys.argv) > 1:
    data_file = sys.argv[1]
    if os.path.exists(data_file):
        # print(f"Файл данных найден: {data_file}")
        pass
    else:
        data_path = os.path.join('data', data_file)
        if os.path.exists(data_path):
            # print(f"Файл данных найден в подкаталоге data: {data_path}")
            pass
        else:
            # print(f"ОШИБКА: Файл данных не найден ни в {data_file}, ни в {data_path}")
            pass

# Парсим аргументы перед импортом TensorFlow и PyTorch
force_cpu = '--force-cpu' in sys.argv
if force_cpu:
    # print("Принудительно используем CPU вместо GPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем GPU для TensorFlow

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    
    # Применяем mixed_precision только если не force_cpu
    if not force_cpu:
        mixed_precision.set_global_policy('mixed_float16')
    # print("TensorFlow успешно загружен")
except Exception as e:
    # print(f"Ошибка при загрузке TensorFlow: {e}")
    traceback.print_exc()

try:
    import io
    import re
    import numpy as np
    import pandas as pd
    import logging
    from tqdm import tqdm, trange
    import matplotlib.pyplot as plt
    # Делаем torch доступным глобально
    global torch
    try:
        import torch
        # Настройка PyTorch для использования CPU
        if force_cpu:
            torch.set_num_threads(os.cpu_count())  # Максимально эффективно используем CPU
        # print("PyTorch успешно загружен")
    except ImportError:
        # print("Ошибка импорта PyTorch")
        pass
    
    # Импортируем модули Stable-Baselines3
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        # print("Stable-Baselines3 успешно загружен")
    except ImportError as e:
        # print(f"Ошибка импорта Stable-Baselines3: {e}")
        sys.exit(1)
    
    # Загружаем модули проекта
    from trading_bot.env import TradingEnv
    from trading_bot.ops import get_state
    from trading_bot.methods import train_model, evaluate_model
    from trading_bot.agent import Agent  # Добавляем импорт класса Agent
    # print("Модули проекта успешно загружены")
except Exception as e:
    # print(f"Ошибка при загрузке библиотек: {e}")
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
    # print("Утилиты успешно загружены")
    
    # Configure multi-core usage - оптимизируем для процессора
    import multiprocessing
    num_physical_cores = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(num_physical_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_physical_cores)
    # Используем все физические ядра для максимальной производительности
    tf.config.threading.set_inter_op_parallelism_threads(num_physical_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_physical_cores)
    # print(f"Многопоточность настроена на {num_physical_cores} ядер")
except Exception as e:
    # print(f"Ошибка при загрузке утилит или настройке многопоточности: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- УБИРАЕМ мусорный DEBUG от PIL НАВСЕГДА! ---
import logging
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def save_window_size(model_path, window_size):
    base = os.path.splitext(model_path)[0]
    fname = base + ".window_size.txt"
    with open(fname, "w") as f:
        f.write(str(window_size))

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=WINDOW_SIZE, replay_freq=100):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    min_v = np.min(data)
    max_v = np.max(data)
    # state_size устанавливается в Agent.__init__, get_state принимает min_v,max_v
    state = get_state(data, 0, window_size, min_v=min_v, max_v=max_v)

    # Создаем tqdm прогресс-бар для шагов внутри эпизода
    progress_bar = tqdm(range(data_length), total=data_length, desc=f'Ep {episode+1}/{ep_count}', position=0, leave=True)
    
    buy_count = 0
    sell_count = 0
    hold_count = 0
    total_reward = 0
    
    for t in progress_bar:
        reward = 0
        next_state = get_state(data, t + 1, window_size, min_v=min_v, max_v=max_v)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            # Проверяем, не превышен ли лимит позиций (максимум 5 позиций)
            if len(agent.inventory) < 5:  # Ограничиваем количество позиций
                agent.inventory.append(float(data[t][0]))
                reward = 2.0
                if len(agent.inventory) <= WINDOW_SIZE:
                    reward += 1.0 * (WINDOW_SIZE - len(agent.inventory))
                buy_count += 1
            else:
                # Если лимит позиций превышен, меняем действие на HOLD
                action = 0
                reward = -5.0  # Штраф за попытку превысить лимит

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = float(data[t][0]) - bought_price
                reward = delta * 200.0 + 10.0
                total_profit += delta
                if delta < 0:
                    reward -= 50.0
                sell_count += 1
            else:
                reward = -20.0
                # Если нечего продавать, меняем действие на HOLD
                action = 0

        # HOLD
        else:
            reward = -1.0 * (t / len(data)) * (len(agent.inventory) + 1)
            if agent.inventory:
                reward -= 0.1 * len(agent.inventory)
            hold_count += 1

        total_reward += reward
        done = (t == data_length - 1)
        
        # Обновляем прогресс-бар с текущей статистикой каждые 100 шагов
        if t % 100 == 0 or t == data_length - 1:
            progress_bar.set_postfix({
                'profit': f'{total_profit:.2f}',
                'reward': f'{total_reward:.2f}',
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count,
                'pos': len(agent.inventory)
            })
            
        agent.remember(state, action, reward, next_state, done)

        # replay every replay_freq steps to speed up training
        if len(agent.memory) > batch_size and t % replay_freq == 0:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # Закрываем прогресс-бар
    progress_bar.close()
    
    # Подробная статистика в конце эпизода
    logging.info(f"Epoch {episode+1}/{ep_count}: profit={total_profit:.2f}, avg_loss={np.mean(np.array(avg_loss)) if avg_loss else 'N/A'}")
    logging.info(f"Actions: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}, Final positions={len(agent.inventory)}")
    logging.info(f"Total reward: {total_reward:.2f}")

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))

def main(stock, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, ep_count=EPISODE_COUNT, 
         strategy="t-dqn", model_name=None, pretrained=False, debug=False, 
         target_update=TARGET_UPDATE, td3_timesteps=TD3_TIMESTEPS, 
         td3_noise_sigma=TD3_NOISE_SIGMA, td3_save_name=TD3_SAVE_NAME):
    """Train an agent."""
    
    # Для TD3 фиксируем window_size=47
    if strategy.lower() == 'td3':
        window_size = WINDOW_SIZE
    
    # Используем глобальные библиотеки
    global pd, np
    # Импортируем torch непосредственно в функции main
    import torch
    
    # Настраиваем логирование
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    
    # Создаем директорию для логов, если её нет
    os.makedirs('logs', exist_ok=True)
    
    # Настраиваем логирование в файл и консоль
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(f"train_{'debug' if debug else 'info'}.log", mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # logger = logging.getLogger(__name__)
    logging.info(f"Запуск обучения: {stock}, strategy={strategy}, window_size={window_size}")
    
    # print("=== ЗАПУСК ФУНКЦИИ MAIN ===")
    # print("Полученные параметры:")
    # print(f"  stock={stock}")
    # print(f"  window_size={window_size}")
    if strategy == "td3":
        # print(f"  strategy={strategy}")
        # print(f"  td3_timesteps={td3_timesteps}")
        # --- Настройка логов ---
        log_file = os.path.join(os.path.dirname(__file__), 'train_finetune.log')
        logging.basicConfig(
            level=logging.INFO,  # Только INFO и выше!
            format='%(asctime)s %(levelname)s %(name)s[%(process)d] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('=== Запуск train.py с нормальными логами ===')
    
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
    # print(f"Train: {len(train_df)} days")
    # print(f"train_data: min={np.min(train_data):.2f}, max={np.max(train_data):.2f}, mean={np.mean(train_data):.2f}")
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
            # print(f'Загрузка дополнительных данных для валидации: {val_file}')
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
            # print(f'Загружены данные для валидации: {len(val_prices)} точек')

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
        
        val_env.reset()
        
        # Создаем модель TD3 с параметрами из конфига
        model = create_TD3_model(train_env, td3_noise_sigma, tb_dir)
        
        # Обучаем модель с визуализацией
        model = train_TD3(model, train_env, val_env, train_env_raw, val_env_raw, 
                         td3_timesteps, td3_save_name, stock, monitor_dir, plots_dir)
        
        return
        
    # --- TQDM TRAIN LOOP ---
    for episode in trange(ep_count, desc='Episodes', position=0, leave=True):
        logging.info(f'START EPISODE {episode + 1}/{ep_count}')
        # update threshold: start aggressive, end conservative
        agent.buy_threshold = -0.01 + (0.01 - (-0.01)) * (episode - 1) / (ep_count - 1)
        result = train_model(agent, episode, train_data, ep_count=ep_count, batch_size=batch_size, window_size=window_size)
        
        # Оценим на трейне векторизованно (batch predict)
        states = np.vstack([get_state(train_data, i, window_size)[0] for i in range(len(train_data)-1)])
        qvals = agent.model.predict(states, verbose=0)
        actions = []
        inventory = []
        
        # Симуляция торговли на основе предсказаний модели
        for q, price in zip(qvals, raw_train_prices[:-1]):
            if q[1] - q[0] > agent.buy_threshold and len(inventory) < 5:  # Ограничиваем количество позиций
                actions.append(1)
                inventory.append(price)
            elif q[2] - q[0] > agent.buy_threshold and inventory:
                actions.append(2)
                inventory.pop(0)
            else:
                actions.append(0)
                
        profit = 0.0
        position = []
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        # Расчет прибыли и статистики
        for i, action in enumerate(actions):
            if action == 1:  # BUY
                buy_count += 1
            elif action == 2:  # SELL
                sell_count += 1
                # Рассчитываем прибыль от продажи
                if i > 0 and len(position) > 0:
                    buy_price = position.pop(0)
                    profit += raw_train_prices[i] - buy_price
            else:  # HOLD
                hold_count += 1
                
            if action == 1:
                position.append(raw_train_prices[i])
                
        # Логируем подробную статистику по эпизоду
        logging.info(f"EPISODE {episode+1} STATS: profit={profit:.2f}, buys={buy_count}, sells={sell_count}, holds={hold_count}")
        logging.info(f"Final positions: {len(position)}, Remaining inventory: {len(agent.inventory)}")
        
        # Оцениваем на валидационных данных
        val_profit, val_history = evaluate_model(agent, val_data, window_size, debug)
        
        # Рассчитываем метрики эффективности
        val_profits = [x[1] for x in val_history if x[1] == 'SELL']
        val_returns = []
        
        # Считаем доходность по каждой сделке
        for i, (price, action) in enumerate(val_history):
            if action == 'SELL' and i > 0:
                for j in range(i-1, -1, -1):
                    if val_history[j][1] == 'BUY':
                        buy_price = val_history[j][0]
                        val_returns.append((price - buy_price) / buy_price)
                        break
                        
        # Рассчитываем метрики
        sharpe_val = np.mean(val_returns) / np.std(val_returns) if val_returns and np.std(val_returns) > 0 else 0
        val_omega = np.sum([r for r in val_returns if r > 0]) / abs(np.sum([r for r in val_returns if r < 0])) if np.sum([r for r in val_returns if r < 0]) != 0 else 0
        val_max_dd = 0
        val_cvar = np.mean([r for r in val_returns if r < 0]) if [r for r in val_returns if r < 0] else 0
        
        # Подробный лог метрик
        logging.info(f"VALIDATION: profit={val_profit:.2f}, sharpe={sharpe_val:.2f}, omega={val_omega:.2f}")
        logging.info(f"Risk metrics: max_drawdown={val_max_dd:.2f}, CVaR={val_cvar:.2f}")
        logging.info(f'END EPISODE {episode + 1}/{ep_count}')
        
    logging.info('=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===')

if __name__ == '__main__':
    # print("Запуск блока __main__")
    try:
        # Исправляем обработку аргументов docopt
        if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
            # Если первый аргумент - CSV файл, то используем его напрямую
            stock = sys.argv[1]
            # print(f"Путь к файлу данных (прямой): {stock}")
        else:
            # Иначе используем docopt
            args = docopt(__doc__)
            # print(f"Аргументы успешно парсинг docopt: {args}")
            stock = args['<stock>']
            # print(f"Путь к файлу данных: {stock}")
        
        # Проверяем существование файла с относительным и абсолютным путем
        if not os.path.isfile(stock):
            # Пробуем искать в подкаталоге data
            data_path = os.path.join('data', stock)
            if os.path.isfile(data_path):
                stock = data_path
                # print(f"Файл найден в подкаталоге data: {stock}")
            else:
                # print(f"Error: Файл {stock} не найден ни в текущем каталоге, ни в подкаталоге data")
                # print(__doc__)
                exit(1)
        if not stock.lower().endswith('.csv'):
            # print(f"Error: Файл должен быть в формате CSV. Получено: {stock}")
            # print(__doc__)
            exit(1)
        # print(f"Используем файл данных: {stock}")
        
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
        
        # print("Все аргументы успешно парсинг")
        
        if force_cpu:
            # print("Режим: принудительное использование CPU")
            pass
        # print(f"ARGS:\n  stock={stock}\n  window_size={window_size}\n  batch_size={batch_size}\n  ep_count={ep_count}\n  strategy={strategy}\n  model_name={model_name}\n  pretrained={pretrained}\n  debug={debug}\n  model_type={model_type}\n  target_update={target_update}\n  td3_timesteps={td3_timesteps}\n  td3_noise_sigma={td3_noise_sigma}\n  td3_save_name={td3_save_name}")
    except Exception as e:
        # print(f"Ошибка при парсинге аргументов: {e}")
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
        window_size = WINDOW_SIZE  # Жёстко фиксируем window_size=47 для TD3
        
    # Определяем имя модели для сохранения
    model_save_name = td3_save_name if strategy.lower() == 'td3' else model_name
    
    # Запускаем основную функцию обучения
    try:
        # print("Запуск main() с параметрами:")
        # print(f"  stock={stock}")
        # print(f"  window_size={window_size}")
        # print(f"  batch_size={batch_size}")
        # print(f"  ep_count={ep_count}")
        # print(f"  strategy={strategy}")
        # print(f"  model_name={model_name}")
        # print(f"  pretrained={pretrained}")
        # print(f"  debug={debug}")
        # print(f"  target_update={target_update}")
        # print(f"  td3_timesteps={td3_timesteps}")
        # print(f"  td3_noise_sigma={td3_noise_sigma}")
        # print(f"  td3_save_name={td3_save_name}")
        
        # Вызываем функцию main с параметрами
        main(stock, window_size, batch_size, ep_count, strategy, model_name, pretrained, debug, target_update, td3_timesteps, td3_noise_sigma, td3_save_name)
        
        # Сохраняем window_size в модель после обучения
        if model_save_name:
            try:
                from stable_baselines3 import TD3
                if os.path.exists(model_save_name):
                    model = TD3.load(model_save_name)
                    model.save(model_save_name, user_data={"window_size": window_size})
                    # print(f"Успешно сохранили window_size={window_size} в модель {model_save_name}")
            except Exception as e:
                # print(f"Ошибка при сохранении window_size в модель: {e}")
                # print(f"[WARNING] Не удалось сохранить window_size в user_data: {e}")
                pass
        
        # print("Скрипт успешно завершен.")
    except Exception as e:
        # print(f"[ERROR] Ошибка в основном блоке: {e}")
        traceback.print_exc()
