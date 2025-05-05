"""
Реализация мультиактивной торговой среды (Часть 2)
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Импортируем класс из первой части
# from multi_asset_environment_part1 import MultiAssetTradingEnv

# Методы для класса MultiAssetTradingEnv
def _get_state(self) -> np.ndarray:
    """
    Формирует вектор состояния для агента согласно статье.
    
    Returns:
        Вектор состояния (наблюдение)
    """
    # Формируем вектор состояния
    state = []
    
    # Добавляем текущий баланс (нормализованный)
    state.append(self.balance / self.initial_balance)
    
    # Для каждого актива добавляем его характеристики
    for i, ticker in enumerate(self.tickers):
        # Получаем текущие данные
        df = self.indicators[ticker]
        current_idx = self.current_step + self.window_size
        
        # Чиним выход за границы индекса
        if current_idx >= len(df):
            current_idx = len(df) - 1
        if current_idx < 0:
            current_idx = 0
        
        # Текущая цена (нормализованная)
        current_price = df['Close'].iloc[current_idx]
        state.append(current_price / df['Close'].iloc[max(0, current_idx-self.window_size):current_idx+1].mean())
        
        # Позиция по активу (нормализованная)
        state.append(self.inventory[i] / self.max_inventory)
        
        # Технические индикаторы (упрощенно для этой части)
        # В реальном использовании здесь будут все индикаторы из статьи
        
        # RSI (если есть)
        if 'RSI' in df.columns:
            state.append(df['RSI'].iloc[current_idx] / 100)
        else:
            state.append(0.5)  # Значение по умолчанию
        
        # MACD (если есть)
        if 'MACD' in df.columns:
            state.append(df['MACD'].iloc[current_idx] / current_price)
            state.append(df['MACD_Signal'].iloc[current_idx] / current_price if 'MACD_Signal' in df.columns else 0)
            state.append(df['MACD_Hist'].iloc[current_idx] / current_price if 'MACD_Hist' in df.columns else 0)
        else:
            state.append(0)
            state.append(0)
            state.append(0)
        
        # Bollinger Bands (если есть)
        if 'BB_Width' in df.columns:
            state.append(df['BB_Width'].iloc[current_idx])
            
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                state.append((current_price - df['BB_Lower'].iloc[current_idx]) / 
                            (df['BB_Upper'].iloc[current_idx] - df['BB_Lower'].iloc[current_idx] + 1e-6))
            else:
                state.append(0.5)
        else:
            state.append(0.1)  # Значение по умолчанию
            state.append(0.5)  # Значение по умолчанию
        
        # ATR (если есть)
        if 'ATR' in df.columns:
            state.append(df['ATR'].iloc[current_idx] / current_price)
        else:
            state.append(0.02)  # Значение по умолчанию
        
        # Объем (если есть)
        if 'Volume_Normalized' in df.columns:
            state.append(df['Volume_Normalized'].iloc[current_idx])
        elif 'Volume' in df.columns:
            # Простая нормализация объема
            state.append(df['Volume'].iloc[current_idx] / (df['Volume'].iloc[max(0, current_idx-self.window_size):current_idx+1].max() + 1e-6))
        else:
            state.append(0.5)  # Значение по умолчанию
        
        # Прибыль/убыток по текущей позиции
        if self.inventory[i] > 0:
            pnl = (current_price - self.avg_buy_price[i]) / self.avg_buy_price[i] if self.avg_buy_price[i] > 0 else 0
        else:
            pnl = 0
        state.append(pnl)
        
        # Максимальная просадка
        state.append(self.max_drawdown)
    
    # Приводим к нужной длине (79)
    needed = 79
    if len(state) < needed:
        state += [0.0] * (needed - len(state))
    elif len(state) > needed:
        state = state[:needed]
    return np.array(state, dtype=np.float32).flatten()

def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    Выполняет один шаг в среде.
    
    Args:
        action: Действие агента (-1 до 1 для каждого актива)
        
    Returns:
        Кортеж (next_state, reward, done, truncated, info)
    """
    # Проверяем, не закончились ли данные
    if self.current_step + self.window_size >= len(next(iter(self.data_dict.values()))) - 1:
        return self._get_state(), 0, True, False, {'balance': self.balance, 'inventory': self.inventory.tolist()}
    
    # Сохраняем предыдущую стоимость портфеля
    prev_portfolio_value = self.portfolio_value
    
    # Обрабатываем действия для каждого актива
    for i, ticker in enumerate(self.tickers):
        # Получаем текущую цену
        df = self.data_dict[ticker]
        current_idx = self.current_step + self.window_size
        # Чиним выход за границы индекса
        if current_idx >= len(df):
            current_idx = len(df) - 1
        if current_idx < 0:
            current_idx = 0
        
        # Интерпретируем действие
        action_value = action[i]  # От -1 до 1
        
        # Определяем количество акций для покупки/продажи
        if action_value > 0:  # Покупка
            # Максимальное количество акций, которое можно купить
            max_buy = min(
                self.balance / (df['Close'].iloc[current_idx] * (1 + self.commission)) / self.max_inventory,
                self.max_inventory - self.inventory[i]
            )
            shares_to_buy = max(0, action_value * max_buy)
            
            # Проверяем минимальный размер сделки
            if shares_to_buy * df['Close'].iloc[current_idx] < self.min_trade_value:
                shares_to_buy = 0
                
            # Выполняем покупку
            if shares_to_buy > 0:
                cost = shares_to_buy * df['Close'].iloc[current_idx] * (1 + self.commission)
                self.balance -= cost
                
                # Обновляем среднюю цену покупки
                if self.inventory[i] == 0:
                    self.avg_buy_price[i] = df['Close'].iloc[current_idx]
                else:
                    self.avg_buy_price[i] = (self.avg_buy_price[i] * self.inventory[i] + df['Close'].iloc[current_idx] * shares_to_buy) / (self.inventory[i] + shares_to_buy)
                
                self.inventory[i] += shares_to_buy
                self.buy_trades[i] += 1
                
        elif action_value < 0:  # Продажа
            shares_to_sell = min(self.inventory[i], -action_value * self.inventory[i])
            
            # Проверяем минимальный размер сделки
            if shares_to_sell * df['Close'].iloc[current_idx] < self.min_trade_value:
                shares_to_sell = 0
                
            # Выполняем продажу
            if shares_to_sell > 0:
                revenue = shares_to_sell * df['Close'].iloc[current_idx] * (1 - self.commission)
                self.balance += revenue
                self.inventory[i] -= shares_to_sell
                self.sell_trades[i] += 1
                
                # Если продали всё, сбрасываем среднюю цену покупки
                if self.inventory[i] == 0:
                    self.avg_buy_price[i] = 0
        
        # Рассчитываем стоимость удержания позиции
        holding_cost = self.inventory[i] * df['Close'].iloc[current_idx] * self.carry_cost
        self.balance -= holding_cost

    # Рассчитываем новую стоимость портфеля
    self.portfolio_value = self.balance + sum([
        self.inventory[i] * self.data_dict[ticker]['Close'].iloc[current_idx]
        for i, ticker in enumerate(self.tickers)
    ])
    
    # Ревард — относительная прибыль (PnL) за шаг
    reward = (self.portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
    
    # Продвигаем шаг
    self.current_step += 1
    done = self.current_step + self.window_size >= len(next(iter(self.data_dict.values()))) - 1
    
    info = {
        'portfolio_value': self.portfolio_value,
        'balance': self.balance,
        'inventory': self.inventory.tolist(),
        'buy_trades': self.buy_trades,
        'sell_trades': self.sell_trades
    }
    return self._get_state(), reward, done, False, info
