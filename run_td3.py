import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from trading_bot.env import TradingEnv
from trading_bot.ops import get_state
import gymnasium as gym

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--window-size", type=int, default=50)
    args = parser.parse_args()

    # load data
    import pandas as pd, io, re
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    raw_lines = []
    with open(args.stock, 'r') as f:
        for line in f:
            if line.startswith('Date,'):
                raw_lines.append(line)
            else:
                m = pattern.search(line)
                if m:
                    raw_lines.append(line[m.start():])
    df = pd.read_csv(io.StringIO('\n'.join(raw_lines)))
    df = df.sort_values('Date').reset_index(drop=True)
    prices = df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2024-01-01')]['Adj Close'].values
    volumes= df[(df['Date'] >= '2015-01-01') & (df['Date'] < '2024-01-01')]['Volume'].values

    env = TradingEnv(prices, volumes, args.window_size)
    n_actions = env.action_space.n
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100000)
    model.save('td3_trading')
