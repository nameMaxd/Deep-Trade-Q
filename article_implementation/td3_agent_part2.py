"""
Реализация TD3 агента согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
Часть 2: Критик и вспомогательные функции
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any, Optional

class TD3Critic(nn.Module):
    """
    Критик для TD3 агента, как описано в статье.
    Принимает состояние и действие, возвращает Q-значение.
    """
    def __init__(self, state_dim: int, action_dim: int, net_arch: List[int] = [256, 256, 128]):
        """
        Инициализация критика.
        
        Args:
            state_dim: Размерность пространства состояний
            action_dim: Размерность пространства действий
            net_arch: Архитектура нейронной сети
        """
        super(TD3Critic, self).__init__()
        
        # Первая Q-сеть
        self.q1_layers = nn.ModuleList()
        
        # Входной слой для Q1
        self.q1_layers.append(nn.Linear(state_dim + action_dim, net_arch[0]))
        
        # Скрытые слои для Q1
        for i in range(len(net_arch) - 1):
            self.q1_layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
        
        # Выходной слой для Q1
        self.q1_output = nn.Linear(net_arch[-1], 1)
        
        # Вторая Q-сеть (для Twin Delayed)
        self.q2_layers = nn.ModuleList()
        
        # Входной слой для Q2
        self.q2_layers.append(nn.Linear(state_dim + action_dim, net_arch[0]))
        
        # Скрытые слои для Q2
        for i in range(len(net_arch) - 1):
            self.q2_layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
        
        # Выходной слой для Q2
        self.q2_output = nn.Linear(net_arch[-1], 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход через обе Q-сети.
        
        Args:
            state: Тензор состояния
            action: Тензор действия
            
        Returns:
            Кортеж из двух Q-значений (q1, q2)
        """
        # Конкатенируем состояние и действие
        sa = torch.cat([state, action], 1)
        
        # Проход через Q1
        q1 = sa
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_output(q1)
        
        # Проход через Q2
        q2 = sa
        for layer in self.q2_layers:
            q2 = F.relu(layer(q2))
        q2 = self.q2_output(q2)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход только через первую Q-сеть.
        
        Args:
            state: Тензор состояния
            action: Тензор действия
            
        Returns:
            Q1-значение
        """
        # Конкатенируем состояние и действие
        sa = torch.cat([state, action], 1)
        
        # Проход через Q1
        q1 = sa
        for layer in self.q1_layers:
            q1 = F.relu(layer(q1))
        q1 = self.q1_output(q1)
        
        return q1

class ReplayBuffer:
    """
    Буфер опыта для TD3 агента.
    Хранит переходы (состояние, действие, награда, следующее состояние, флаг завершения).
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        """
        Инициализация буфера опыта.
        
        Args:
            state_dim: Размерность пространства состояний
            action_dim: Размерность пространства действий
            max_size: Максимальный размер буфера
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Буферы для хранения переходов
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool):
        """
        Добавляет переход в буфер.
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            next_state: Следующее состояние
            reward: Полученная награда
            done: Флаг завершения эпизода
        """
        # Гарантируем, что state и next_state — одномерные массивы правильной длины
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.ndim > 1:
            state = state.flatten()
        if next_state.ndim > 1:
            next_state = next_state.flatten()
        if state.shape[0] != self.state.shape[1]:
            raise ValueError(f"state.shape {state.shape} не совпадает с размерностью буфера {self.state.shape[1]}")
        if next_state.shape[0] != self.next_state.shape[1]:
            raise ValueError(f"next_state.shape {next_state.shape} не совпадает с размерностью буфера {self.next_state.shape[1]}")
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        
        # Обновляем указатель и размер
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Выбирает случайный батч из буфера.
        
        Args:
            batch_size: Размер батча
            
        Returns:
            Кортеж из тензоров (состояния, действия, следующие состояния, награды, флаги не завершения)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
