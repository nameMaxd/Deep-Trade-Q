"""
Реализация торговой среды как Partially Observed Markov Decision Process (POMDP)
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Any, Optional
import matplotlib.pyplot as plt

class TradingEnvArticle(gym.Env):
    """
    Торговая среда, реализующая POMDP модель из статьи.
    Поддерживает торговлю несколькими активами с непрерывным пространством действий.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 window_size: int = 47,
                 initial_balance: float = 10000.0,
                 commission: float = 0.001,
                 carry_cost: float = 0.0001,
                 min_trade_value: float = 1000.0,
                 max_inventory: int = 8,
                 risk_lambda: float = 0.1,
                 drawdown_lambda: float = 0.1,
                 volume_bins: int = 10,
                 volume_lookback: int = 50):
        """
        Инициализация торговой среды.
        
        Args:
            df: DataFrame с историческими данными
            window_size: Размер окна для анализа (количество баров)
            initial_balance: Начальный депозит
            commission: Комиссия за сделку (в процентах)
            carry_cost: Стоимость удержания позиции
            min_trade_value: Минимальный размер сделки
            max_inventory: Максимальное количество позиций
            risk_lambda: Коэффициент штрафа за риск
            drawdown_lambda: Коэффициент штрафа за просадку
            volume_bins: Количество уровней объема для анализа
            volume_lookback: Глубина анализа объема (баров)
        """
        super(TradingEnvArticle, self).__init__()
        
        # Сохраняем параметры
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.carry_cost = carry_cost
        self.min_trade_value = min_trade_value
        self.max_inventory = max_inventory
        self.risk_lambda = risk_lambda
        self.drawdown_lambda = drawdown_lambda
        self.volume_bins = volume_bins
        self.volume_lookback = volume_lookback
        
        # Рассчитываем количество активов (в статье поддерживается мультиактивная торговля)
        # В нашем случае торгуем одним активом, но код готов к расширению
        self.num_assets = 1
        
        # Определяем пространство состояний: (1 + 13 x N)-мерный вектор, где N - количество активов
        # 1 - для баланса, 13 - для каждого актива (цена, технические индикаторы и т.д.)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1 + 13 * self.num_assets,),
            dtype=np.float32
        )
        
        # Определяем пространство действий: непрерывное от -1 до 1 для каждого актива
        # -1 означает максимальную продажу, 1 - максимальную покупку
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        # Инициализация переменных состояния
        self.reset()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Рассчитывает технические индикаторы, упомянутые в статье.
        
        Args:
            data: DataFrame с историческими данными
            
        Returns:
            Словарь с рассчитанными индикаторами
        """
        # Копируем данные, чтобы не изменять оригинал
        df = data.copy()
        
        # Рассчитываем RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Рассчитываем MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_hist = macd - signal
        
        # Рассчитываем Bollinger Bands
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        bb_width = (upper_band - lower_band) / sma20
        
        # Рассчитываем ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Рассчитываем OBV (On-Balance Volume)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Рассчитываем Volume Profile
        # Упрощенная версия - просто нормализуем объем
        volume_normalized = df['Volume'] / df['Volume'].rolling(window=self.volume_lookback).max()
        
        # Возвращаем словарь с индикаторами
        return {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': macd_hist,
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_width': bb_width,
            'atr': atr,
            'obv': obv,
            'volume_normalized': volume_normalized
        }
    
    def _get_state(self) -> np.ndarray:
        """
        Формирует вектор состояния для агента согласно статье.
        
        Returns:
            Вектор состояния (наблюдение)
        """
        # Получаем текущие данные в окне
        end_idx = self.current_step + self.window_size
        if end_idx > len(self.df):
            end_idx = len(self.df)
        
        window_data = self.df.iloc[self.current_step:end_idx]
        
        # Рассчитываем технические индикаторы
        indicators = self._calculate_technical_indicators(window_data)
        
        # Формируем вектор состояния
        state = []
        
        # Добавляем текущий баланс (нормализованный)
        state.append(self.balance / self.initial_balance)
        
        # Для каждого актива добавляем его характеристики
        for _ in range(self.num_assets):
            # Текущая цена (нормализованная)
            current_price = window_data['Close'].iloc[-1]
            state.append(current_price / window_data['Close'].mean())
            
            # Позиция по активу (нормализованная)
            state.append(self.inventory / self.max_inventory if self.inventory else 0)
            
            # Технические индикаторы
            state.append(indicators['rsi'].iloc[-1] / 100)  # RSI нормализован от 0 до 1
            state.append(indicators['macd'].iloc[-1] / current_price)  # MACD нормализован относительно цены
            state.append(indicators['macd_signal'].iloc[-1] / current_price)
            state.append(indicators['macd_hist'].iloc[-1] / current_price)
            state.append(indicators['bb_width'].iloc[-1])  # BB Width уже нормализован
            state.append((current_price - indicators['bb_lower'].iloc[-1]) / 
                        (indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]))  # Позиция цены в BB
            state.append(indicators['atr'].iloc[-1] / current_price)  # ATR нормализован относительно цены
            state.append(indicators['volume_normalized'].iloc[-1])  # Объем уже нормализован
            
            # Прибыль/убыток по текущей позиции
            if self.inventory > 0:
                pnl = (current_price - self.avg_buy_price) / self.avg_buy_price
            else:
                pnl = 0
            state.append(pnl)
            
            # Максимальная просадка
            state.append(self.max_drawdown)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Выполняет один шаг в среде.
        
        Args:
            action: Действие агента (-1 до 1 для каждого актива)
            
        Returns:
            Кортеж (next_state, reward, done, truncated, info)
        """
        # Проверяем, не закончились ли данные
        if self.current_step + self.window_size >= len(self.df) - 1:
            return self._get_state(), 0, True, False, {'balance': self.balance, 'inventory': self.inventory}
        
        # Получаем текущую цену
        current_price = self.df['Close'].iloc[self.current_step + self.window_size]
        
        # Интерпретируем действие для первого актива
        action_value = action[0]  # От -1 до 1
        
        # Определяем количество акций для покупки/продажи
        if action_value > 0:  # Покупка
            # Максимальное количество акций, которое можно купить
            max_buy = min(
                self.balance / (current_price * (1 + self.commission)) / self.max_inventory,
                self.max_inventory - self.inventory
            )
            shares_to_buy = max(0, action_value * max_buy)
            
            # Проверяем минимальный размер сделки
            if shares_to_buy * current_price < self.min_trade_value:
                shares_to_buy = 0
                
            # Выполняем покупку
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.commission)
                self.balance -= cost
                
                # Обновляем среднюю цену покупки
                if self.inventory == 0:
                    self.avg_buy_price = current_price
                else:
                    self.avg_buy_price = (self.avg_buy_price * self.inventory + current_price * shares_to_buy) / (self.inventory + shares_to_buy)
                
                self.inventory += shares_to_buy
                self.buy_trades += 1
                
        elif action_value < 0:  # Продажа
            shares_to_sell = min(self.inventory, -action_value * self.inventory)
            
            # Проверяем минимальный размер сделки
            if shares_to_sell * current_price < self.min_trade_value:
                shares_to_sell = 0
                
            # Выполняем продажу
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.commission)
                self.balance += revenue
                self.inventory -= shares_to_sell
                self.sell_trades += 1
                
                # Если продали всё, сбрасываем среднюю цену покупки
                if self.inventory == 0:
                    self.avg_buy_price = 0
        
        # Рассчитываем стоимость удержания позиции
        holding_cost = self.inventory * current_price * self.carry_cost
        self.balance -= holding_cost
        
        # Рассчитываем текущую стоимость портфеля
        portfolio_value = self.balance + (self.inventory * current_price)
        
        # Обновляем максимальную стоимость портфеля и максимальную просадку
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Рассчитываем вознаграждение
        # 1. Изменение стоимости портфеля
        portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        
        # 2. Штраф за риск (волатильность)
        risk_penalty = self.risk_lambda * self.df['Close'].iloc[self.current_step:self.current_step + self.window_size].std() / current_price
        
        # 3. Штраф за просадку
        drawdown_penalty = self.drawdown_lambda * self.max_drawdown
        
        # Итоговое вознаграждение
        reward = portfolio_return - risk_penalty - drawdown_penalty
        
        # Обновляем предыдущую стоимость портфеля
        self.prev_portfolio_value = portfolio_value
        
        # Переходим к следующему шагу
        self.current_step += 1
        
        # Формируем информацию для отладки
        info = {
            'balance': self.balance,
            'inventory': self.inventory,
            'portfolio_value': portfolio_value,
            'portfolio_return': portfolio_return,
            'risk_penalty': risk_penalty,
            'drawdown_penalty': drawdown_penalty,
            'max_drawdown': self.max_drawdown,
            'buy_trades': self.buy_trades,
            'sell_trades': self.sell_trades
        }
        
        return self._get_state(), reward, False, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Сбрасывает среду в начальное состояние.
        
        Args:
            seed: Seed для генератора случайных чисел
            options: Дополнительные опции
            
        Returns:
            Кортеж (initial_state, info)
        """
        super().reset(seed=seed)
        
        # Инициализация переменных состояния
        self.balance = self.initial_balance
        self.inventory = 0
        self.avg_buy_price = 0
        self.current_step = 0
        self.max_portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        self.buy_trades = 0
        self.sell_trades = 0
        
        return self._get_state(), {'balance': self.balance, 'inventory': self.inventory}
    
    def render(self, mode='human'):
        """
        Визуализирует текущее состояние среды.
        
        Args:
            mode: Режим визуализации
        """
        if mode != 'human':
            return
        
        # Получаем данные для визуализации
        end_idx = min(self.current_step + self.window_size, len(self.df) - 1)
        plot_data = self.df.iloc[max(0, end_idx - 100):end_idx + 1]
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        
        # График цены
        plt.subplot(3, 1, 1)
        plt.plot(plot_data['Close'], label='Close Price')
        plt.title('Trading Environment')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # График объема
        plt.subplot(3, 1, 2)
        plt.bar(range(len(plot_data)), plot_data['Volume'], color='blue', alpha=0.5)
        plt.ylabel('Volume')
        plt.grid(True)
        
        # График портфеля
        plt.subplot(3, 1, 3)
        portfolio_values = []
        for i in range(max(0, end_idx - 100), end_idx + 1):
            portfolio_value = self.balance + (self.inventory * self.df['Close'].iloc[i])
            portfolio_values.append(portfolio_value)
        
        plt.plot(portfolio_values, label='Portfolio Value', color='green')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """
        Закрывает среду и освобождает ресурсы.
        """
        plt.close('all')
