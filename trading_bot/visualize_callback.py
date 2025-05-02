import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

class VisualizeCallback(BaseCallback):
    """
    Callback for plotting and saving buy/sell markers on price chart for train and validation environments.
    Generates up to `max_plots` evenly spaced evaluation plots.
    """
    def __init__(self, train_env, val_env, model, total_timesteps, max_plots=100, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.val_env = val_env
        self.model = model
        self.max_plots = max_plots
        # compute eval interval
        self.eval_freq = max(1, total_timesteps // max_plots)
        # prepare folders
        self.plots_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(os.path.join(self.plots_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'val'), exist_ok=True)
        self.plot_count = 0

    def _on_step(self) -> bool:
        # trigger at eval frequency
        if self.n_calls % self.eval_freq == 0:
            self.plot_count += 1
            # create train/val plots
            try:
                self._make_plot(self.train_env, os.path.join(self.plots_dir, 'train', f'plot_{self.plot_count}.png'))
            except Exception as e:
                print(f"[VisualizeCallback] Ошибка при построении train plot: {e}")
            try:
                self._make_plot(self.val_env,   os.path.join(self.plots_dir, 'val',   f'plot_{self.plot_count}.png'))
            except Exception as e:
                print(f"[VisualizeCallback] Ошибка при построении val plot: {e}")
            # create trades table for validation
            try:
                trades_csv = os.path.join(self.plots_dir, 'val', f'trades_{self.plot_count}.csv')
                self._make_table(self.val_env, trades_csv)
            except Exception as e:
                print(f"[VisualizeCallback] Ошибка при построении trades table: {e}")
        return True

    def _make_plot(self, env, save_path):
        # gather price and actions
        real_env = getattr(env, 'env', env)  # поддержка Monitor
        prices = list(getattr(real_env, 'prices', []))
        obs, _ = env.reset()
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        step = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            price = prices[real_env.current_step] if prices else None
            if info.get('real_action') == 1:
                buy_x.append(step); buy_y.append(price)
            elif info.get('real_action') == 2:
                sell_x.append(step); sell_y.append(price)
            step += 1
            if done:
                break
        # plot
        plt.figure(figsize=(10, 4))
        if prices:
            plt.plot(prices[:step], label='price')
        if buy_x:
            plt.scatter(buy_x, buy_y, marker='^', color='g', label='BUY')
        if sell_x:
            plt.scatter(sell_x, sell_y, marker='v', color='r', label='SELL')
        plt.legend(loc='best')
        plt.title(os.path.basename(save_path))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _make_table(self, env, table_path):
        """Simulate env and save validation trades table with buy/sell prices and profit pct"""
        prices = list(env.prices)
        obs, _ = env.reset()
        history = []
        pending = []
        step = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            price = prices[env.current_step]
            if info.get('real_action') == 1:
                pending.append((step, price))
            elif info.get('real_action') == 2 and pending:
                buy_step, buy_price = pending.pop(0)
                sell_step, sell_price = step, price
                profit_pct = (sell_price - buy_price) / buy_price * 100
                history.append({
                    'buy_step': buy_step,
                    'buy_price': buy_price,
                    'sell_step': sell_step,
                    'sell_price': sell_price,
                    'profit_pct': profit_pct
                })
            step += 1
            if done:
                break
        df = pd.DataFrame(history)
        df.to_csv(table_path, index=False)
