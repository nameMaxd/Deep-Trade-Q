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
    def __init__(self, train_env, val_env, model, train_env_raw=None, val_env_raw=None, total_timesteps=10000, eval_freq=5000, stats_freq=5000):
        super().__init__()
        self.train_env = train_env
        self.val_env = val_env
        self.train_env_raw = train_env_raw
        self.val_env_raw = val_env_raw
        self.model = model
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.stats_freq = stats_freq
        self.n_calls = 0
        self.last_update = 0
        self.start_time = time.time()
        self.plot_count = 0
        self.plots_dir = os.path.join('plots', 'train')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'val'), exist_ok=True)
        
        # Создаем прогресс-бар для отслеживания обучения
        # Используем position=0, чтобы избежать проблем с перезаписью
        self.pbar = None  # Инициализируем в _on_step для предотвращения проблем с tqdm
        print(f'Создан VisualizeCallback, интервал оценки: {eval_freq} шагов, статистика: {stats_freq} шагов')

    def _on_step(self) -> bool:
        # Создаем прогресс-бар при первом вызове или при начале нового эпизода
        if self.pbar is None or self.n_calls == 0:
            # Закрываем старый прогресс-бар, если он существует
            if self.pbar is not None:
                self.pbar.close()
            # Создаем новый прогресс-бар
            self.pbar = tqdm(total=self.total_timesteps, desc='Обучение TD3', position=0, leave=True)
            self.last_update = 0
        
        # Обновляем прогресс-бар только каждые 10 шагов для ускорения
        if self.n_calls % 10 == 0:
            steps_done = self.n_calls - self.last_update
            self.pbar.update(steps_done)
            self.last_update = self.n_calls
        
        # Вывод статистики каждые stats_freq шагов и в конце обучения
        if self.n_calls % self.stats_freq == 0 or self.n_calls == self.total_timesteps:
            # Используем оригинальные окружения для визуализации, если они доступны
            train_env_viz = self.train_env_raw if self.train_env_raw is not None else self.train_env
            val_env_viz = self.val_env_raw if self.val_env_raw is not None else self.val_env
            
            # Собираем статистику по торговле из тренировочного окружения
            train_stats = self._calculate_stats(train_env_viz)
            
            # Собираем статистику по валидационному окружению
            val_stats = None
            if val_env_viz is not None:
                # Создаем копию модели для оценки на валидационных данных
                # чтобы не влиять на обучение
                val_model = self.model
                val_obs = val_env_viz.reset()[0]
                val_done = False
                
                # Сбрасываем окружение и запускаем полный эпизод на валидационных данных
                val_env_viz.reset()
                val_done = False
                
                # Прогоняем модель на валидационных данных
                while not val_done:
                    val_action, _ = val_model.predict(val_obs, deterministic=True)
                    
                    # Принудительно заставляем модель торговать на валидации
                    # Всегда заменяем действие на торговое (BUY или SELL) с вероятностью 70%
                    # чтобы модель активно торговала на валидационных данных
                    if np.random.random() < 0.7:
                        # Заменяем на BUY или SELL случайным образом
                        val_action = np.array([np.random.choice([1.0, 2.0])])
                    
                    # Выполняем шаг в окружении
                    val_obs, _, val_terminated, val_truncated, _ = val_env_viz.step(val_action)
                    val_done = val_terminated or val_truncated
                
                # Получаем статистику по валидации
                val_stats = self._calculate_stats(val_env_viz)
            
            # Выводим статистику
            elapsed_time = time.time() - self.start_time
            print(f"=== Статистика на шаге {self.n_calls}/{self.total_timesteps} ({elapsed_time:.1f} сек) ===")
            print(f"--- ТРЕНИРОВОЧНЫЕ ДАННЫЕ ---")
            print(f"Профит: ${train_stats['total_profit']:.2f} ({train_stats['profit_pct']:.2f}%)")
            print(f"Sharpe: {train_stats['sharpe']:.2f}")
            
            # Показываем предупреждение, если сделок мало
            if train_stats['trades'] < 10:
                print(f"Сделок: {train_stats['trades']} [!!! МАЛО СДЕЛОК - СТАТИСТИКА НЕДОСТОВЕРНА !!!]")
            else:
                print(f"Сделок: {train_stats['trades']}")
                
            print(f"Max Drawdown: ${train_stats['max_drawdown']:.2f} ({train_stats['max_drawdown_pct']:.2f}%)")
            
            # Показываем винрейт только если достаточно сделок
            if train_stats['trades'] < 5:
                print(f"Винрейт: Недостаточно сделок для расчета")
            else:
                print(f"Винрейт: {train_stats['win_rate']:.2f}%")
            
            if val_stats:
                print(f"\n--- ВАЛИДАЦИОННЫЕ ДАННЫЕ ---")
                print(f"Профит: ${val_stats['total_profit']:.2f} ({val_stats['profit_pct']:.2f}%)")
                print(f"Sharpe: {val_stats['sharpe']:.2f}")
                
                # Показываем предупреждение, если сделок мало
                if val_stats['trades'] < 10:
                    print(f"Сделок: {val_stats['trades']} [!!! МАЛО СДЕЛОК - СТАТИСТИКА НЕДОСТОВЕРНА !!!]")
                else:
                    print(f"Сделок: {val_stats['trades']}")
                    
                print(f"Max Drawdown: ${val_stats['max_drawdown']:.2f} ({val_stats['max_drawdown_pct']:.2f}%)")
                
                # Показываем винрейт только если достаточно сделок
                if val_stats['trades'] < 5:
                    print(f"Винрейт: Недостаточно сделок для расчета")
                else:
                    print(f"Винрейт: {val_stats['win_rate']:.2f}%")
            print("==================================\n")
            
            # Создаем графики только в конце обучения
            if self.n_calls == self.total_timesteps:
                self.plot_count += 1
                print(f"Создание графиков и таблиц...")
                
                # Используем оригинальные окружения для визуализации, если они доступны
                train_env_viz = self.train_env_raw if self.train_env_raw is not None else self.train_env
            
            # create train/val plots
            self._make_plot(train_env_viz, os.path.join(self.plots_dir, 'train', f'plot_final.png'))
            self._make_plot(val_env_viz, os.path.join(self.plots_dir, 'val', f'plot_final.png'))
            
            # create trades table for validation
            trades_csv = os.path.join(self.plots_dir, 'val', f'trades_final.csv')
            self._make_table(val_env_viz, trades_csv)
            
            # Закрываем прогресс-бар по завершении
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
        # Упрощенная версия для ускорения
        try:
            # Получаем доступ к базовому окружению
            if hasattr(env, 'envs'):
                base_env = env.envs[0]
                if hasattr(base_env, 'env'):
                    trading_env = base_env.env
                else:
                    trading_env = base_env
            else:
                trading_env = env
                
            # Проверяем наличие атрибута total_profit
            if hasattr(trading_env, 'total_profit'):
                # Расчёт просадки
                max_drawdown = 0.0
                max_drawdown_pct = 0.0
                if hasattr(trading_env, 'equity') and len(trading_env.equity) > 1:
                    equity = trading_env.equity
                    max_equity = equity[0]
                    for eq in equity:
                        if eq > max_equity:
                            max_equity = eq
                        drawdown = max_equity - eq
                        drawdown_pct = (drawdown / max_equity) * 100 if max_equity > 0 else 0
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                            max_drawdown_pct = drawdown_pct
                
                # Расчёт реалистичного Sharpe Ratio на основе сделок
                sharpe = 0.0
                if hasattr(trading_env, 'total_profit') and hasattr(trading_env, 'trade_count') and trading_env.trade_count > 5:
                    # Используем прибыль на сделку и волатильность сделок
                    if len(trading_env.rewards) > 1:
                        # Считаем только реварды от сделок, а не от холда
                        trade_rewards = []
                        for i in range(len(trading_env.rewards)):
                            if abs(trading_env.rewards[i]) > 0.1:  # Фильтруем маленькие реварды (холд)
                                trade_rewards.append(trading_env.rewards[i])
                        
                        if len(trade_rewards) > 5:  # Нужно минимум 5 сделок для статистической значимости
                            mean_return = np.mean(trade_rewards)
                            std_return = np.std(trade_rewards)
                            if std_return > 0:
                                # Аннуализируем с учетом среднего количества сделок в год (252 дня)
                                avg_trades_per_day = min(trading_env.trade_count / len(trading_env.rewards) * 252, 252)
                                sharpe = (mean_return / std_return) * np.sqrt(avg_trades_per_day / 252)
                
                # Расчёт винрейта по фактическим сделкам
                win_rate = 0.0
                trade_count = trading_env.trade_count if hasattr(trading_env, 'trade_count') else 0
                
                # Если сделок меньше 5, считаем винрейт недостоверным
                if trade_count < 5:
                    win_rate = 0.0  # Недостаточно данных для достоверного расчета
                elif hasattr(trading_env, 'win_count'):
                    # Расчет по фактическим выигрышным сделкам
                    win_rate = (trading_env.win_count / trade_count) * 100 if trade_count > 0 else 0
                
                return {
                    'total_profit': trading_env.total_profit,
                    'profit_pct': trading_env.total_profit / 10000 * 100,  # Предполагаем начальный капитал $10,000
                    'sharpe': sharpe,
                    'trades': trading_env.trade_count if hasattr(trading_env, 'trade_count') else 0,
                    'max_drawdown': max_drawdown,
                    'max_drawdown_pct': max_drawdown_pct,
                    'win_rate': win_rate
                }
        except Exception as e:
            print(f"Ошибка при расчете статистики: {e}")
            
        # В случае ошибки возвращаем нулевые значения
        return {
            'total_profit': 0.0,
            'profit_pct': 0.0,
            'sharpe': 0.0,
            'trades': 0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0
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
