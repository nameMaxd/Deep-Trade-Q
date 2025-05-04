"""
Реализация мультиактивной торговой среды (Часть 3)
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from multi_asset_environment_part2 import _get_state, step

# Продолжение методов для класса MultiAssetTradingEnv

def step_continued(self, action: np.ndarray, prev_portfolio_value=None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    Продолжение метода step из части 2.
    """
    # Рассчитываем текущую стоимость портфеля
    self.portfolio_value = self.balance
    for i, ticker in enumerate(self.tickers):
        df = self.data_dict[ticker]
        current_idx = self.current_step + self.window_size
        # Чиним выход за границы индекса
        if current_idx >= len(df):
            current_idx = len(df) - 1
        if current_idx < 0:
            current_idx = 0
        current_price = df['Close'].iloc[current_idx]
        self.portfolio_value += self.inventory[i] * current_price
    
    # Обновляем максимальную стоимость портфеля и максимальную просадку
    if self.portfolio_value > self.max_portfolio_value:
        self.max_portfolio_value = self.portfolio_value
    
    current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
    self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    # Рассчитываем вознаграждение
    # 1. Изменение стоимости портфеля
    if prev_portfolio_value is None:
        prev_portfolio_value = self.portfolio_value
    portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
    
    # 2. Штраф за риск (волатильность)
    risk_penalty = 0
    for ticker in self.tickers:
        df = self.data_dict[ticker]
        current_idx = self.current_step + self.window_size
        # Чиним выход за границы индекса
        if current_idx >= len(df):
            current_idx = len(df) - 1
        if current_idx < 0:
            current_idx = 0
        current_price = df['Close'].iloc[current_idx]
        
        # Рассчитываем волатильность цены
        price_volatility = df['Close'].iloc[max(0, current_idx-self.window_size):current_idx+1].std() / current_price
        risk_penalty += self.risk_lambda * price_volatility
    
    # Нормализуем штраф за риск
    risk_penalty /= self.num_assets
    
    # 3. Штраф за просадку
    drawdown_penalty = self.drawdown_lambda * self.max_drawdown
    
    # Итоговое вознаграждение
    reward = portfolio_return - risk_penalty - drawdown_penalty
    
    # Переходим к следующему шагу
    self.current_step += 1
    
    # Формируем информацию для отладки
    info = {
        'balance': self.balance,
        'inventory': self.inventory.tolist(),
        'portfolio_value': self.portfolio_value,
        'portfolio_return': portfolio_return,
        'risk_penalty': risk_penalty,
        'drawdown_penalty': drawdown_penalty,
        'max_drawdown': self.max_drawdown,
        'buy_trades': self.buy_trades.tolist(),
        'sell_trades': self.sell_trades.tolist()
    }
    
    return self._get_state(), reward, False, False, info

def render(self, mode='human'):
    """
    Визуализирует текущее состояние среды.
    
    Args:
        mode: Режим визуализации
    """
    if mode != 'human':
        return
    
    # Получаем данные для визуализации
    end_idx = min(self.current_step + self.window_size, len(next(iter(self.data_dict.values()))) - 1)
    start_idx = max(0, end_idx - 100)
    
    # Создаем график
    fig, axes = plt.subplots(2 + self.num_assets, 1, figsize=(15, 5 * (2 + self.num_assets)))
    
    # График стоимости портфеля
    portfolio_values = []
    for i in range(start_idx, end_idx + 1):
        portfolio_value = self.balance
        for j, ticker in enumerate(self.tickers):
            df = self.data_dict[ticker]
            portfolio_value += self.inventory[j] * df['Close'].iloc[i]
        portfolio_values.append(portfolio_value)
    
    axes[0].plot(range(start_idx, end_idx + 1), portfolio_values, label='Portfolio Value', color='green')
    axes[0].set_title('Portfolio Value')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    axes[0].legend()
    
    # График распределения активов
    asset_values = []
    for j, ticker in enumerate(self.tickers):
        df = self.data_dict[ticker]
        asset_values.append(self.inventory[j] * df['Close'].iloc[end_idx])
    
    # Добавляем баланс
    asset_values.append(self.balance)
    labels = self.tickers + ['Cash']
    
    axes[1].pie(asset_values, labels=labels, autopct='%1.1f%%')
    axes[1].set_title('Asset Allocation')
    
    # Графики цен активов
    for i, ticker in enumerate(self.tickers):
        df = self.data_dict[ticker]
        plot_data = df.iloc[start_idx:end_idx+1]
        
        axes[i+2].plot(range(start_idx, end_idx + 1), plot_data['Close'], label=f'{ticker} Price', color='blue')
        axes[i+2].set_title(f'{ticker} Price')
        axes[i+2].set_ylabel('Price')
        axes[i+2].grid(True)
        axes[i+2].legend()
        
        # Добавляем индикаторы, если они есть
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            axes[i+2].plot(range(start_idx, end_idx + 1), plot_data['BB_Upper'], label='BB Upper', color='red', alpha=0.5)
            axes[i+2].plot(range(start_idx, end_idx + 1), plot_data['BB_Lower'], label='BB Lower', color='red', alpha=0.5)
            axes[i+2].fill_between(range(start_idx, end_idx + 1), plot_data['BB_Upper'], plot_data['BB_Lower'], color='gray', alpha=0.2)
        
        # Отмечаем позиции
        for j in range(start_idx, end_idx + 1):
            if j > 0 and self.inventory[i] > self.inventory_history[i][j-1]:
                axes[i+2].scatter(j, plot_data['Close'].iloc[j-start_idx], color='green', marker='^', s=100)
            elif j > 0 and self.inventory[i] < self.inventory_history[i][j-1]:
                axes[i+2].scatter(j, plot_data['Close'].iloc[j-start_idx], color='red', marker='v', s=100)
    
    plt.tight_layout()
    plt.show()

def close(self):
    """
    Закрывает среду и освобождает ресурсы.
    """
    plt.close('all')

# Дополнительные методы для работы с мультиактивной средой

def combine_environments(self):
    """
    Метод для объединения методов из разных частей.
    """
    # Добавляем методы из части 2
    self._get_state = _get_state.__get__(self)
    # Объединяем методы step и step_continued
    original_step = step.__get__(self)
    continued_step = step_continued.__get__(self)
    
    def combined_step(action):
        prev_portfolio_value = self.portfolio_value
        original_step(action)
        return continued_step(action, prev_portfolio_value)
    
    self.step = combined_step
    
    # Добавляем методы из части 3
    self.render = render.__get__(self)
    self.close = close.__get__(self)
    
    # Инициализируем историю инвентаря для визуализации
    self.inventory_history = [[] for _ in range(self.num_assets)]
    
    # Модифицируем метод reset для сброса истории инвентаря
    original_reset = self.reset
    
    def reset_with_history(*args, **kwargs):
        state, info = original_reset(*args, **kwargs)
        self.inventory_history = [[] for _ in range(self.num_assets)]
        return state, info
    
    self.reset = reset_with_history
    
    # Модифицируем метод step для обновления истории инвентаря
    original_combined_step = self.step
    
    def step_with_history(action):
        next_state, reward, done, truncated, info = original_combined_step(action)
        for i in range(self.num_assets):
            self.inventory_history[i].append(self.inventory[i])
        return next_state, reward, done, truncated, info
    
    self.step = step_with_history
    
    return self
