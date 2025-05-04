"""
Реализация технических индикаторов для торговой системы
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

class TechnicalIndicators:
    """
    Класс для расчета технических индикаторов, упомянутых в статье.
    """
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Рассчитывает индикатор RSI (Relative Strength Index).
        
        Args:
            prices: Массив цен
            period: Период для расчета
            
        Returns:
            Массив значений RSI
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else np.inf
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else np.inf
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Рассчитывает индикатор MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Массив цен
            fast_period: Период для быстрой EMA
            slow_period: Период для медленной EMA
            signal_period: Период для сигнальной линии
            
        Returns:
            Кортеж (macd, signal, histogram)
        """
        # Рассчитываем EMA
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow_period)
        
        # Рассчитываем MACD
        macd = ema_fast - ema_slow
        
        # Рассчитываем сигнальную линию
        signal = TechnicalIndicators.calculate_ema(macd, signal_period)
        
        # Рассчитываем гистограмму
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Рассчитывает EMA (Exponential Moving Average).
        
        Args:
            prices: Массив цен
            period: Период для расчета
            
        Returns:
            Массив значений EMA
        """
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        multiplier = 2.0 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Рассчитывает полосы Боллинджера.
        
        Args:
            prices: Массив цен
            period: Период для расчета
            num_std: Количество стандартных отклонений
            
        Returns:
            Кортеж (upper_band, middle_band, lower_band)
        """
        # Рассчитываем SMA
        middle_band = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < period:
                middle_band[i] = np.mean(prices[:i+1])
            else:
                middle_band[i] = np.mean(prices[i-period+1:i+1])
        
        # Рассчитываем стандартное отклонение
        std = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < period:
                std[i] = np.std(prices[:i+1])
            else:
                std[i] = np.std(prices[i-period+1:i+1])
        
        # Рассчитываем верхнюю и нижнюю полосы
        upper_band = middle_band + num_std * std
        lower_band = middle_band - num_std * std
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Рассчитывает ATR (Average True Range).
        
        Args:
            high: Массив максимальных цен
            low: Массив минимальных цен
            close: Массив цен закрытия
            period: Период для расчета
            
        Returns:
            Массив значений ATR
        """
        # Рассчитываем True Range
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Рассчитываем ATR
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        
        for i in range(1, len(tr)):
            if i < period:
                atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    @staticmethod
    def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Рассчитывает OBV (On-Balance Volume).
        
        Args:
            close: Массив цен закрытия
            volume: Массив объемов
            
        Returns:
            Массив значений OBV
        """
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    def calculate_volume_profile(volume: np.ndarray, close: np.ndarray, bins: int = 10, lookback: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Рассчитывает профиль объема.
        
        Args:
            volume: Массив объемов
            close: Массив цен закрытия
            bins: Количество уровней
            lookback: Глубина анализа
            
        Returns:
            Кортеж (volume_levels, volume_values)
        """
        volume_profile = np.zeros((len(close), bins))
        
        for i in range(len(close)):
            if i < lookback:
                continue
            
            # Получаем данные за период lookback
            period_close = close[i-lookback:i]
            period_volume = volume[i-lookback:i]
            
            # Определяем границы уровней
            min_price = np.min(period_close)
            max_price = np.max(period_close)
            
            if min_price == max_price:
                continue
            
            price_range = max_price - min_price
            level_size = price_range / bins
            
            # Распределяем объем по уровням
            for j in range(lookback):
                level = int((period_close[j] - min_price) / level_size)
                if level >= bins:
                    level = bins - 1
                
                volume_profile[i, level] += period_volume[j]
        
        # Нормализуем профиль объема
        volume_profile_normalized = np.zeros_like(volume_profile)
        for i in range(len(close)):
            if np.sum(volume_profile[i]) > 0:
                volume_profile_normalized[i] = volume_profile[i] / np.sum(volume_profile[i])
        
        return volume_profile_normalized
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает все технические индикаторы для DataFrame.
        
        Args:
            df: DataFrame с данными (должен содержать столбцы Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # Копируем DataFrame
        result = df.copy()
        
        # Рассчитываем RSI
        result['RSI'] = TechnicalIndicators.calculate_rsi(result['Close'].values)
        
        # Рассчитываем MACD
        macd, signal, histogram = TechnicalIndicators.calculate_macd(result['Close'].values)
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Hist'] = histogram
        
        # Рассчитываем полосы Боллинджера
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(result['Close'].values)
        result['BB_Upper'] = upper
        result['BB_Middle'] = middle
        result['BB_Lower'] = lower
        result['BB_Width'] = (upper - lower) / middle
        
        # Рассчитываем ATR
        result['ATR'] = TechnicalIndicators.calculate_atr(
            result['High'].values,
            result['Low'].values,
            result['Close'].values
        )
        
        # Рассчитываем OBV
        result['OBV'] = TechnicalIndicators.calculate_obv(
            result['Close'].values,
            result['Volume'].values
        )
        
        # Нормализуем объем
        result['Volume_Normalized'] = result['Volume'] / result['Volume'].rolling(window=50).max()
        
        return result
