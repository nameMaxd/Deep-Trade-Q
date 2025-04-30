import numpy as np
import pandas as pd
import io
import re
import sys
import os
from stable_baselines3 import TD3
from trading_bot.env import TradingEnv

MODEL_PATH = r"td3_model_data/GOOG_2010-2024-06.zip"
DATA_PATH = r"data/GOOG_2024-07_2025-04.csv"
WINDOW_SIZE = 50  # Можно изменить при необходимости


def load_prices_volumes(csv_path, start_date=None, end_date=None):
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    raw_lines = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Date,'):
                raw_lines.append(line)
            else:
                m = pattern.search(line)
                if m:
                    raw_lines.append(line[m.start():])
    df = pd.read_csv(io.StringIO('\n'.join(raw_lines)))
    df = df.sort_values('Date').reset_index(drop=True)
    if start_date and end_date:
        prices = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]['Adj Close'].values
        volumes = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]['Volume'].values
    else:
        prices = df['Adj Close'].values
        volumes = df['Volume'].values
    return prices, volumes


def evaluate_td3(model_path, data_path, window_size):
    prices, volumes = load_prices_volumes(data_path)
    env = TradingEnv(prices, volumes, window_size)
    print(f"[eval] state_size={env.state_size} obs_shape={env.observation_space.shape}")
    model = TD3.load(model_path)
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
    if getattr(env, 'inventory', None):
        final_price = float(env.prices[env.current_step])
        for bought_price, qty in env.inventory:
            total_profit += (final_price - bought_price) * qty
    print(f"TD3 EVAL PROFIT: {total_profit:.2f}, TRADES: {trades}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=MODEL_PATH)
    parser.add_argument('--data-path', type=str, default=DATA_PATH)
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    args = parser.parse_args()
    evaluate_td3(args.model_path, args.data_path, args.window_size)
