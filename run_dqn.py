import re
import io
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from trading_bot.env import TradingEnv

def load_data(path, window_size):
    # Clean CSV: keep header and valid date lines
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    raw_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Date,'):
                raw_lines.append(line)
            else:
                m = pattern.search(line)
                if m:
                    raw_lines.append(line[m.start():])
    df = pd.read_csv(io.StringIO("\n".join(raw_lines)))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Split: train 2015-2023, val 2024-01 to 2024-06
    train_df = df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2024-01-01')]
    val_df   = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-06-30')]
    return (
        train_df['Adj Close'].values, train_df['Volume'].values,
        val_df['Adj Close'].values,   val_df['Volume'].values
    )

def evaluate(env, model):
    obs, _ = env.reset()
    done = False
    total_profit = 0.0
    trades = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if action != 0:
            trades += 1
        obs, reward, done, _, _ = env.step(action)
        total_profit += reward
    # liquidate
    if hasattr(env, 'inventory') and env.inventory:
        final_price = env.prices[env.current_step]
        for bp in env.inventory:
            total_profit += final_price - bp
    return total_profit, trades

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--timesteps', type=int, default=500000)
    args = parser.parse_args()

    train_p, train_v, val_p, val_v = load_data(args.stock, args.window_size)
    train_env = TradingEnv(train_p, train_v, args.window_size)
    val_env   = TradingEnv(val_p, val_v, args.window_size)

    # DQN hyperparameters tuned for discrete trading
    model = DQN(
        'MlpPolicy', train_env,
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=64,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=1000,
        train_freq=1,
        verbose=1
    )
    # Evaluation callback on val set
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path='./dqn_best',
        log_path='./dqn_logs',
        eval_freq=50_000,
        deterministic=True
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save('dqn_trading')

    # Final evaluation
    pt, tt = evaluate(TradingEnv(train_p, train_v, args.window_size), model)
    pv, tv = evaluate(TradingEnv(val_p, val_v, args.window_size), model)
    print(f"DQN Train Profit: {pt:.4f}, Trades: {tt}")
    print(f"DQN Val   Profit: {pv:.4f}, Trades: {tv}")
