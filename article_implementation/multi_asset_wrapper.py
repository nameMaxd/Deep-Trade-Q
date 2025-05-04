"""
Обертка для объединения всех частей мультиактивной торговой среды
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym

# Импортируем классы и методы из разных частей
from multi_asset_environment_part1 import MultiAssetTradingEnv
from multi_asset_environment_part2 import _get_state, step
from multi_asset_environment_part3 import step_continued, render, close, combine_environments

# Добавляем методы к классу MultiAssetTradingEnv
MultiAssetTradingEnv._get_state = _get_state
MultiAssetTradingEnv._step_part1 = step
MultiAssetTradingEnv._step_part2 = step_continued
MultiAssetTradingEnv.render = render
MultiAssetTradingEnv.close = close
MultiAssetTradingEnv.combine_environments = combine_environments

def create_multi_asset_env(
    data_dict: Dict[str, pd.DataFrame],
    window_size: int = 47,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    carry_cost: float = 0.0001,
    min_trade_value: float = 1000.0,
    max_inventory: int = 8,
    risk_lambda: float = 0.1,
    drawdown_lambda: float = 0.1,
    volume_bins: int = 10,
    volume_lookback: int = 50,
    sentiment_data: Optional[Dict[str, pd.DataFrame]] = None
) -> MultiAssetTradingEnv:
    """
    Создает и настраивает мультиактивную торговую среду.
    
    Args:
        data_dict: Словарь с данными для каждого актива {ticker: DataFrame}
        window_size: Размер окна для анализа
        initial_balance: Начальный баланс
        commission: Комиссия за сделку
        carry_cost: Стоимость удержания позиции
        min_trade_value: Минимальный размер сделки
        max_inventory: Максимальное количество позиций для каждого актива
        risk_lambda: Коэффициент штрафа за риск
        drawdown_lambda: Коэффициент штрафа за просадку
        volume_bins: Количество уровней объема
        volume_lookback: Глубина анализа объема
        sentiment_data: Словарь с данными о настроениях для каждого актива
        
    Returns:
        Настроенная мультиактивная торговая среда
    """
    # Создаем среду
    env = MultiAssetTradingEnv(
        data_dict=data_dict,
        window_size=window_size,
        initial_balance=initial_balance,
        commission=commission,
        carry_cost=carry_cost,
        min_trade_value=min_trade_value,
        max_inventory=max_inventory,
        risk_lambda=risk_lambda,
        drawdown_lambda=drawdown_lambda,
        volume_bins=volume_bins,
        volume_lookback=volume_lookback,
        sentiment_data=sentiment_data
    )
    
    # Объединяем методы из разных частей
    env = env.combine_environments()
    
    # Переопределяем метод step для корректной работы
    original_step_part1 = env._step_part1
    original_step_part2 = env._step_part2
    
    def combined_step(action):
        # Сохраняем предыдущую стоимость портфеля для расчета вознаграждения
        prev_portfolio_value = env.portfolio_value
        
        # Вызываем первую часть step
        original_step_part1(action)
        
        # Вызываем вторую часть step с передачей prev_portfolio_value
        return original_step_part2(action)
    
    env.step = combined_step
    
    # Инициализируем историю инвентаря для визуализации
    env.inventory_history = [[] for _ in range(env.num_assets)]
    
    # Модифицируем метод reset для сброса истории инвентаря
    original_reset = env.reset
    
    def reset_with_history(*args, **kwargs):
        state, info = original_reset(*args, **kwargs)
        env.inventory_history = [[] for _ in range(env.num_assets)]
        return state, info
    
    env.reset = reset_with_history
    
    # Модифицируем метод step для обновления истории инвентаря
    original_combined_step = env.step
    
    def step_with_history(action):
        next_state, reward, done, truncated, info = original_combined_step(action)
        for i in range(env.num_assets):
            env.inventory_history[i].append(env.inventory[i])
        return next_state, reward, done, truncated, info
    
    env.step = step_with_history
    
    return env
