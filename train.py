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

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    WINDOW_SIZE,
    minmax_normalize,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
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

    # Настраиваем логирование в файл
    logging.basicConfig(filename="train_finetune.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("Лог обучения будет писаться в train_finetune.log")

    # Загружаем данные 2019-01-01 — 2024-06-30
    # Clean CSV: keep only header and lines starting with date
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    raw_lines = []
    with open(stock, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Date,'):
                raw_lines.append(line)
            else:
                m = pattern.search(line)
                if m:
                    raw_lines.append(line[m.start():])
    df = pd.read_csv(io.StringIO('\n'.join(raw_lines)))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Split into train (2015–2023) and validation (2024-01-01–2024-06-30)
    train_df = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2024-01-01")]
    val_df = df[(df["Date"] >= "2024-01-01") & (df["Date"] <= "2024-06-30")]
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
        # Wrap training env with Monitor for logging rewards
        train_env = Monitor(train_env, monitor_dir)
        n_actions = train_env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
        # Initialize TD3 with TensorBoard logging
        model = TD3('MlpPolicy', train_env, action_noise=action_noise, verbose=1,
                    tensorboard_log=tb_dir)
        # attach visualization callback
        visual_cb = VisualizeCallback(train_env, val_env, model, total_timesteps=td3_timesteps)
        model.learn(total_timesteps=td3_timesteps, callback=visual_cb)
        model.save(f"{td3_save_name}")
        # Plot training reward progression from Monitor logs
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(os.path.join(monitor_dir, 'monitor.csv'), comment='#')
        plt.figure(figsize=(8,4))
        plt.plot(df['l'], df['r'], label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward Progression')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_reward.png'))
        plt.close()
        return
    # если выбран TD3, запускаем Stable-Baselines3 TD3
    if strategy.lower() == 'td3':
        from trading_bot.env import TradingEnv
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.callbacks import BaseCallback
        from tqdm import tqdm
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
        model = TD3(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            action_noise=action_noise
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
        try:
            model.learn(total_timesteps=td3_timesteps, callback=[tqdm_cb, eval_cb])
        except KeyboardInterrupt:
            print("Training interrupted by user, proceeding to final inference...")
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
    # Resolve pretrained model path: normalize to basename
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
