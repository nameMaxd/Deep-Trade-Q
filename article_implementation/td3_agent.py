"""
Реализация TD3 агента согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
Часть 1: Основные классы и функции
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any, Optional
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class TD3Actor(nn.Module):
    """
    Актор для TD3 агента, как описано в статье.
    Принимает состояние и возвращает действие.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, net_arch: List[int] = [256, 256, 128]):
        """
        Инициализация актора.
        
        Args:
            state_dim: Размерность пространства состояний
            action_dim: Размерность пространства действий
            max_action: Максимальное значение действия
            net_arch: Архитектура нейронной сети
        """
        super(TD3Actor, self).__init__()
        
        # Создаем слои нейронной сети
        self.layers = nn.ModuleList()
        
        # Входной слой
        self.layers.append(nn.Linear(state_dim, net_arch[0]))
        
        # Скрытые слои
        for i in range(len(net_arch) - 1):
            self.layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
        
        # Выходной слой
        self.output_layer = nn.Linear(net_arch[-1], action_dim)
        
        self.max_action = max_action
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через нейронную сеть.
        
        Args:
            state: Тензор состояния
            
        Returns:
            Тензор действия
        """
        x = state
        
        # Проходим через все слои, кроме выходного
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Выходной слой с tanh для ограничения действий в диапазоне [-1, 1]
        x = self.max_action * torch.tanh(self.output_layer(x))
        
        return x
