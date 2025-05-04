"""
Реализация профиля объема для торговой системы
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

class VolumeProfile:
    """
    Класс для расчета и визуализации профиля объема, как описано в статье.
    """
    
    def __init__(self, bins: int = 10, lookback: int = 50):
        """
        Инициализирует профиль объема.
        
        Args:
            bins: Количество уровней
            lookback: Глубина анализа
        """
        self.bins = bins
        self.lookback = lookback
    
    def calculate_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает профиль объема для DataFrame.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Close и Volume)
            
        Returns:
            DataFrame с добавленными уровнями объема
        """
        # Копируем DataFrame
        result = df.copy()
        
        # Получаем массивы цен и объемов
        close = result['Close'].values
        volume = result['Volume'].values
        
        # Рассчитываем профиль объема
        volume_profile = np.zeros((len(close), self.bins))
        
        for i in range(len(close)):
            if i < self.lookback:
                continue
            
            # Получаем данные за период lookback
            period_close = close[i-self.lookback:i]
            period_volume = volume[i-self.lookback:i]
            
            # Определяем границы уровней
            min_price = np.min(period_close)
            max_price = np.max(period_close)
            
            if min_price == max_price:
                continue
            
            price_range = max_price - min_price
            level_size = price_range / self.bins
            
            # Распределяем объем по уровням
            for j in range(self.lookback):
                level = int((period_close[j] - min_price) / level_size)
                if level >= self.bins:
                    level = self.bins - 1
                
                volume_profile[i, level] += period_volume[j]
        
        # Нормализуем профиль объема
        volume_profile_normalized = np.zeros_like(volume_profile)
        for i in range(len(close)):
            if np.sum(volume_profile[i]) > 0:
                volume_profile_normalized[i] = volume_profile[i] / np.sum(volume_profile[i])
        
        # Добавляем уровни объема в DataFrame
        for i in range(self.bins):
            result[f'Volume_Level_{i}'] = volume_profile_normalized[:, i]
        
        return result
    
    def get_support_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """
        Определяет уровни поддержки и сопротивления на основе профиля объема.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Close и Volume)
            window: Окно для анализа
            
        Returns:
            Кортеж (уровни поддержки, уровни сопротивления)
        """
        # Рассчитываем профиль объема
        volume_df = self.calculate_profile(df)
        
        # Получаем текущие данные
        current_data = volume_df.iloc[-window:]
        
        # Определяем границы цен
        min_price = current_data['Close'].min()
        max_price = current_data['Close'].max()
        price_range = max_price - min_price
        level_size = price_range / self.bins
        
        # Получаем средний профиль объема за окно
        volume_levels = np.zeros(self.bins)
        for i in range(self.bins):
            volume_levels[i] = current_data[f'Volume_Level_{i}'].mean()
        
        # Определяем уровни поддержки и сопротивления
        support_levels = []
        resistance_levels = []
        
        # Находим локальные максимумы
        for i in range(1, self.bins - 1):
            if volume_levels[i] > volume_levels[i-1] and volume_levels[i] > volume_levels[i+1]:
                # Рассчитываем цену для уровня
                level_price = min_price + (i + 0.5) * level_size
                
                # Определяем, поддержка это или сопротивление
                if level_price < current_data['Close'].iloc[-1]:
                    support_levels.append(level_price)
                else:
                    resistance_levels.append(level_price)
        
        return support_levels, resistance_levels
    
    def visualize_profile(self, df: pd.DataFrame, window: int = 50):
        """
        Визуализирует профиль объема.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Close и Volume)
            window: Окно для визуализации
        """
        # Рассчитываем профиль объема
        volume_df = self.calculate_profile(df)
        
        # Получаем текущие данные
        current_data = volume_df.iloc[-window:]
        
        # Определяем границы цен
        min_price = current_data['Close'].min()
        max_price = current_data['Close'].max()
        price_range = max_price - min_price
        level_size = price_range / self.bins
        
        # Получаем средний профиль объема за окно
        volume_levels = np.zeros(self.bins)
        for i in range(self.bins):
            volume_levels[i] = current_data[f'Volume_Level_{i}'].mean()
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})
        
        # График цены
        ax1.plot(current_data.index, current_data['Close'], label='Close Price')
        ax1.set_title('Price Chart')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Определяем уровни поддержки и сопротивления
        support_levels, resistance_levels = self.get_support_resistance_levels(df, window)
        
        # Добавляем уровни поддержки и сопротивления на график
        for level in support_levels:
            ax1.axhline(y=level, color='g', linestyle='--', alpha=0.7)
        
        for level in resistance_levels:
            ax1.axhline(y=level, color='r', linestyle='--', alpha=0.7)
        
        # График профиля объема
        price_levels = np.linspace(min_price, max_price, self.bins)
        ax2.barh(price_levels, volume_levels, height=level_size, alpha=0.7)
        ax2.set_title('Volume Profile')
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Price')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

class LiquidityAnalyzer:
    """
    Класс для анализа ликвидности, как описано в статье.
    """
    
    def __init__(self, volume_profile: VolumeProfile):
        """
        Инициализирует анализатор ликвидности.
        
        Args:
            volume_profile: Объект для расчета профиля объема
        """
        self.volume_profile = volume_profile
    
    def calculate_liquidity_score(self, df: pd.DataFrame, window: int = 20) -> float:
        """
        Рассчитывает оценку ликвидности.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Close и Volume)
            window: Окно для анализа
            
        Returns:
            Оценка ликвидности
        """
        # Получаем текущие данные
        current_data = df.iloc[-window:]
        
        # Рассчитываем средний объем
        avg_volume = current_data['Volume'].mean()
        
        # Рассчитываем волатильность
        volatility = current_data['Close'].std() / current_data['Close'].mean()
        
        # Рассчитываем спред (упрощенно, как разницу между High и Low)
        avg_spread = (current_data['High'] - current_data['Low']).mean() / current_data['Close'].mean()
        
        # Рассчитываем оценку ликвидности
        # Высокий объем, низкая волатильность и низкий спред = высокая ликвидность
        liquidity_score = (avg_volume / df['Volume'].mean()) / (volatility * avg_spread)
        
        return liquidity_score
    
    def get_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Получает признаки ликвидности для DataFrame.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Close, High, Low и Volume)
            
        Returns:
            DataFrame с добавленными признаками ликвидности
        """
        # Копируем DataFrame
        result = df.copy()
        
        # Рассчитываем оценку ликвидности
        result['Liquidity_Score'] = np.nan
        
        for i in range(20, len(result)):
            result.loc[result.index[i], 'Liquidity_Score'] = self.calculate_liquidity_score(result.iloc[:i+1])
        
        # Нормализуем оценку ликвидности
        max_score = result['Liquidity_Score'].max()
        if max_score > 0:
            result['Liquidity_Score_Normalized'] = result['Liquidity_Score'] / max_score
        else:
            result['Liquidity_Score_Normalized'] = 0
        
        # Добавляем другие признаки ликвидности
        result['Volume_MA'] = result['Volume'].rolling(window=20).mean() / result['Volume'].mean()
        result['Spread'] = (result['High'] - result['Low']) / result['Close']
        result['Spread_MA'] = result['Spread'].rolling(window=20).mean()
        
        return result
