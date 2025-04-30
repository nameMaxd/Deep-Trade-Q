import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import time
import pandas as pd

class VisualizeCallback(BaseCallback):
    """
    Callback for plotting and saving buy/sell markers on price chart for train and validation environments.
    Generates up to `max_plots` evenly spaced evaluation plots.
    """
    def __init__(self, train_env, val_env, model, train_env_raw=None, val_env_raw=None, total_timesteps=100000, max_plots=100, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env  # Векторизованное окружение для обучения
        self.val_env = val_env    # Векторизованное окружение для валидации
        self.train_env_raw = train_env_raw  # Оригинальное окружение для визуализации
        self.val_env_raw = val_env_raw    # Оригинальное окружение для визуализации
        self.model = model
        self.max_plots = max_plots
        self.total_timesteps = total_timesteps
        # Графики и таблицы каждые 5000 шагов
        self.eval_freq = 5000
        # Статистика каждые 5000 шагов
        self.stats_freq = 5000
        # prepare folders
        self.plots_dir = os.path.join(os.getcwd(), 'plots')
        os.makedirs(os.path.join(self.plots_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'val'), exist_ok=True)
        self.plot_count = 0
        
        # Инициализация прогресс-бара
        self.pbar = tqdm(total=total_timesteps, desc='Обучение TD3')
        self.last_update = 0
        self.start_time = time.time()
        
        print(f"Создан VisualizeCallback, интервал оценки: {self.eval_freq} шагов, статистика: {self.stats_freq} шагов")

    def _on_step(self) -> bool:
        # Обновляем прогресс-бар
        steps_done = self.n_calls - self.last_update
        self.pbar.update(steps_done)
        self.last_update = self.n_calls
        
        # Вывод статистики каждые stats_freq шагов
        if self.n_calls % self.stats_freq == 0 or self.n_calls == self.total_timesteps:
            # Используем оригинальные окружения для визуализации, если они доступны
            val_env_viz = self.val_env_raw if self.val_env_raw is not None else self.val_env
            
            # Собираем статистику по торговле
            stats = self._calculate_stats(val_env_viz)
            
            # Выводим статистику
            elapsed_time = time.time() - self.start_time
            print(f"\n=== Статистика на {self.n_calls} шаге ({elapsed_time:.1f} сек) ===")
            print(f"Профит: ${stats['total_profit']:.2f} ({stats['profit_pct']:.2f}%)")
            print(f"Sharpe: {stats['sharpe']:.2f}")
            print(f"Сделок: {stats['trades']}")
            print(f"Max Drawdown: ${stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']:.2f}%)")
            print(f"Винрейт: {stats['win_rate']:.2f}%")
            print("==================================\n")
        
        # trigger at eval frequency
        if self.n_calls % self.eval_freq == 0:
            self.plot_count += 1
            print(f"Создание графиков и таблиц #{self.plot_count}...")
            
            # Используем оригинальные окружения для визуализации, если они доступны
            train_env_viz = self.train_env_raw if self.train_env_raw is not None else self.train_env
            val_env_viz = self.val_env_raw if self.val_env_raw is not None else self.val_env
            
            # create train/val plots
            self._make_plot(train_env_viz, os.path.join(self.plots_dir, 'train', f'plot_{self.plot_count}.png'))
            self._make_plot(val_env_viz, os.path.join(self.plots_dir, 'val', f'plot_{self.plot_count}.png'))
            
            # create trades table for validation
            trades_csv = os.path.join(self.plots_dir, 'val', f'trades_{self.plot_count}.csv')
            self._make_table(val_env_viz, trades_csv)
            
        # Закрываем прогресс-бар по завершении
        if self.n_calls >= self.total_timesteps:
            self.pbar.close()
            
        return True

    def _make_plot(self, env, save_path):
        # Проверяем, является ли окружение векторизованным
        if hasattr(env, 'venv'):
            # Если это векторизованное окружение, берем первый элемент
            base_env = env.envs[0]
        else:
            base_env = env
            
        # Проверяем, является ли окружение объектом Monitor
        if hasattr(base_env, 'env'):
            # Если это Monitor, то доступаемся к внутреннему окружению
            trading_env = base_env.env
        else:
            trading_env = base_env
            
        # Проверяем, что это TradingEnv
        if not hasattr(trading_env, 'prices'):
            print(f"Ошибка: не найден атрибут 'prices' в {type(trading_env)}")
            return
            
        # gather price and actions
        prices = list(trading_env.prices)
        obs = env.reset()
        # Если возвращается кортеж, берем первый элемент
        if isinstance(obs, tuple):
            obs = obs[0]
            
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        step = 0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Если информация в векторизованном формате, берем первый элемент
            if isinstance(info, list):
                info = info[0]
                
            # Получаем текущий шаг и цену
            current_step = min(step, len(prices)-1)  # Защита от выхода за границы
            price = prices[current_step]
            
            if info.get('real_action') == 1:
                buy_x.append(step); buy_y.append(price)
            elif info.get('real_action') == 2:
                sell_x.append(step); sell_y.append(price)
                
            obs = next_obs
            done = terminated or truncated
            step += 1
            
            # Защита от бесконечного цикла
            if step > len(prices):
                break
        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(prices[:step], label='price')
        plt.scatter(buy_x, buy_y, marker='^', color='g', label='BUY')
        plt.scatter(sell_x, sell_y, marker='v', color='r', label='SELL')
        plt.legend(loc='best')
        plt.title(os.path.basename(save_path))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _calculate_stats(self, env):
        """Рассчитывает статистику торговли: профит, шарп, макс просадку и т.д."""
        # Проверяем, является ли окружение векторизованным
        if hasattr(env, 'venv'):
            # Если это векторизованное окружение, берем первый элемент
            base_env = env.envs[0]
        else:
            base_env = env
            
        # Проверяем, является ли окружение объектом Monitor
        if hasattr(base_env, 'env'):
            # Если это Monitor, то доступаемся к внутреннему окружению
            trading_env = base_env.env
        else:
            trading_env = base_env
            
        # Проверяем, что это TradingEnv
        if not hasattr(trading_env, 'prices'):
            print(f"Ошибка: не найден атрибут 'prices' в {type(trading_env)}")
            return {
                'total_profit': 0.0,
                'profit_pct': 0.0,
                'sharpe': 0.0,
                'trades': 0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0
            }
            
        prices = list(trading_env.prices)
        obs = env.reset()
        # Если возвращается кортеж, берем первый элемент
        if isinstance(obs, tuple):
            obs = obs[0]
            
        # История сделок
        history = []
        # Открытые позиции
        pending = []
        # Эквити кривая (баланс по шагам)
        equity = [10000.0]  # Начальный капитал $10,000
        # Текущий баланс
        balance = equity[0]
        
        step = 0
        done = False
        
        try:
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Если информация в векторизованном формате, берем первый элемент
                if isinstance(info, list):
                    info = info[0]
                    
                # Получаем текущий шаг и цену
                current_step = min(step, len(prices)-1)  # Защита от выхода за границы
                price = prices[current_step]
                
                # Обработка действий
                if info.get('real_action') == 1:  # Покупка
                    # Условно покупаем акции на фиксированную сумму $1000
                    shares = 1000 / price
                    cost = shares * price
                    balance -= cost
                    pending.append((step, price, shares))
                    
                elif info.get('real_action') == 2 and pending:  # Продажа
                    buy_step, buy_price, shares = pending.pop(0)
                    sell_step, sell_price = step, price
                    
                    # Расчет прибыли/убытка
                    revenue = shares * sell_price
                    profit = revenue - (shares * buy_price)
                    profit_pct = (sell_price - buy_price) / buy_price * 100
                    
                    # Обновляем баланс
                    balance += revenue
                    
                    # Записываем сделку
                    history.append({
                        'buy_step': buy_step,
                        'buy_price': buy_price,
                        'sell_step': sell_step,
                        'sell_price': sell_price,
                        'shares': shares,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                
                # Обновляем эквити
                equity.append(balance + sum([shares * price for _, price, shares in pending]))
                
                obs = next_obs
                done = terminated or truncated
                step += 1
                
                # Защита от бесконечного цикла
                if step > len(prices):
                    break
        except Exception as e:
            print(f"Ошибка при расчете статистики: {e}")
            
        # Расчет статистики
        if not history:
            return {
                'total_profit': 0.0,
                'profit_pct': 0.0,
                'sharpe': 0.0,
                'trades': 0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0
            }
            
        # Общая прибыль
        total_profit = sum([trade['profit'] for trade in history])
        profit_pct = (equity[-1] - equity[0]) / equity[0] * 100
        
        # Расчет просадки
        max_equity = equity[0]
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        
        for eq in equity:
            if eq > max_equity:
                max_equity = eq
            drawdown = max_equity - eq
            drawdown_pct = (drawdown / max_equity) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # Расчет Sharpe Ratio (упрощенно)
        daily_returns = [(equity[i] - equity[i-1]) / equity[i-1] for i in range(1, len(equity))]
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)  # Аннуализированный
        else:
            sharpe = 0.0
            
        # Винрейт
        winning_trades = sum(1 for trade in history if trade['profit'] > 0)
        win_rate = (winning_trades / len(history)) * 100 if history else 0.0
        
        return {
            'total_profit': total_profit,
            'profit_pct': profit_pct,
            'sharpe': sharpe,
            'trades': len(history),
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate': win_rate
        }
        
    def _make_table(self, env, table_path):
        """Simulate env and save validation trades table with buy/sell prices and profit pct"""
        # Проверяем, является ли окружение векторизованным
        if hasattr(env, 'venv'):
            # Если это векторизованное окружение, берем первый элемент
            base_env = env.envs[0]
        else:
            base_env = env
            
        # Проверяем, является ли окружение объектом Monitor
        if hasattr(base_env, 'env'):
            # Если это Monitor, то доступаемся к внутреннему окружению
            trading_env = base_env.env
        else:
            trading_env = base_env
            
        # Проверяем, что это TradingEnv
        if not hasattr(trading_env, 'prices'):
            print(f"Ошибка: не найден атрибут 'prices' в {type(trading_env)}")
            return
            
        prices = list(trading_env.prices)
        obs = env.reset()
        # Если возвращается кортеж, берем первый элемент
        if isinstance(obs, tuple):
            obs = obs[0]
            
        history = []
        pending = []
        step = 0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Если информация в векторизованном формате, берем первый элемент
            if isinstance(info, list):
                info = info[0]
                
            # Получаем текущий шаг и цену
            current_step = min(step, len(prices)-1)  # Защита от выхода за границы
            price = prices[current_step]
            
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
                
            obs = next_obs
            done = terminated or truncated
            step += 1
            
            # Защита от бесконечного цикла
            if step > len(prices):
                break
        df = pd.DataFrame(history)
        df.to_csv(table_path, index=False)
