import numpy as np
import pandas as pd
import io
import re
import os
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from trading_bot.env import TradingEnv

MODEL_PATH = r"td3_model_data/GOOG_2010-2024-06.zip"
DATA_PATH = r"data/GOOG_2024-07_2025-04.csv"

# --- Автоматическая загрузка window_size из файла рядом с моделью ---
def load_window_size(model_path):
    import os
    base = os.path.splitext(model_path)[0]
    for ext in [".window_size.txt", ".window_size", ".meta.txt", ".meta"]:
        fname = base + ext
        if os.path.exists(fname):
            with open(fname, "r") as f:
                return int(f.read().strip())
    raise RuntimeError(f"Не найден файл window_size рядом с моделью: {base}.window_size.txt. Сохрани window_size при обучении!")

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

def evaluate_td3_viz(model_path, data_path, window_size=None, plot_path=None):
    from stable_baselines3 import TD3
    model = TD3.load(model_path)
    if window_size is None:
        # Пытаемся взять window_size из user_data модели
        try:
            window_size = model.user_data["window_size"]
            print(f"[INFO] window_size взят из user_data модели: {window_size}")
        except Exception:
            raise RuntimeError("window_size не передан и не найден в user_data модели!")
    prices, volumes = load_prices_volumes(data_path)
    env = TradingEnv(prices, volumes, window_size)
    print(f"[DEBUG] window_size={window_size}, state_size={env.state_size}, obs_space={env.observation_space.shape}")
    obs, _ = env.reset()
    print(f"[DEBUG] obs.shape={obs.shape}")
    if obs.shape[0] != 53:
        raise RuntimeError(f"ОШИБКА: state_dim+2={obs.shape[0]}, а модель ожидает 53! ПОДБЕРИ window_size так, чтобы state_dim+2=53!")
    done = False
    total_profit = 0.0
    trades = 0
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    steps, rews = [], []
    sharpe_list = []
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        steps.append(env.current_step)
        rews.append(reward)
        if info.get('real_action') == 1:
            buy_x.append(env.current_step)
            buy_y.append(prices[env.current_step])
            trades += 1
        elif info.get('real_action') == 2:
            sell_x.append(env.current_step)
            sell_y.append(prices[env.current_step])
            trades += 1
        total_profit += reward
    if getattr(env, 'inventory', None):
        final_price = float(env.prices[env.current_step])
        for bought_price, qty in env.inventory:
            total_profit += (final_price - bought_price) * qty
    rews = np.array(rews)
    sharpe = (rews.mean() / (rews.std() + 1e-8)) * np.sqrt(252) if len(rews) > 1 else 0
    pct_profit = (env.total_profit / prices[0]) * 100 if prices[0] != 0 else 0
    print(f"TD3 EVAL PROFIT: {total_profit:.2f}, TRADES: {trades}, SHARPE: {sharpe:.3f}, PROFIT %: {pct_profit:.2f}")
    if plot_path is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price')
        plt.scatter(buy_x, buy_y, marker='^', color='g', label='BUY')
        plt.scatter(sell_x, sell_y, marker='v', color='r', label='SELL')
        plt.title(f"TD3 Evaluation\nProfit: {total_profit:.2f}, Trades: {trades}, Sharpe: {sharpe:.3f}, Profit %: {pct_profit:.2f}")
        plt.xlabel('Step')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    return total_profit, trades, sharpe, pct_profit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=MODEL_PATH)
    parser.add_argument('--data-path', type=str, default=DATA_PATH)
    parser.add_argument('--window-size', type=int, default=None, help='Если не указано, берётся из файла рядом с моделью')
    parser.add_argument('--plot-path', type=str, default='eval_td3_plot.png')
    args = parser.parse_args()
    evaluate_td3_viz(args.model_path, args.data_path, args.window_size, args.plot_path)
