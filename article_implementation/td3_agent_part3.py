"""
Реализация TD3 агента согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
Часть 3: Основной класс TD3Agent
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Any, Optional
import copy

# Импортируем классы из предыдущих файлов
# В реальном использовании нужно импортировать из td3_agent.py и td3_agent_part2.py
from td3_agent import TD3Actor
from td3_agent_part2 import TD3Critic, ReplayBuffer

class TD3Agent:
    """
    Реализация TD3 агента согласно статье.
    Twin Delayed Deep Deterministic Policy Gradient (TD3) - улучшенная версия DDPG.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        net_arch: List[int] = [256, 256, 128]
    ):
        """
        Инициализация TD3 агента.
        
        Args:
            state_dim: Размерность пространства состояний
            action_dim: Размерность пространства действий
            max_action: Максимальное значение действия
            discount: Коэффициент дисконтирования
            tau: Коэффициент мягкого обновления целевых сетей
            policy_noise: Шум для сглаживания целевой политики
            noise_clip: Ограничение шума политики
            policy_freq: Частота обновления политики (в шагах)
            lr_actor: Скорость обучения актора
            lr_critic: Скорость обучения критика
            net_arch: Архитектура нейронной сети
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Инициализируем актора
        self.actor = TD3Actor(state_dim, action_dim, max_action, net_arch).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Инициализируем критика
        self.critic = TD3Critic(state_dim, action_dim, net_arch).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Инициализируем буфер опыта
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        
        # Сохраняем гиперпараметры
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Счетчик обновлений
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, add_noise: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        """
        Выбирает действие на основе текущего состояния.
        
        Args:
            state: Текущее состояние
            add_noise: Добавлять ли шум для исследования
            noise_scale: Масштаб шума
            
        Returns:
            Выбранное действие
        """
        # Преобразуем состояние в тензор
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # Получаем действие от актора
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        # Добавляем шум для исследования, если нужно
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def predict(self, obs, deterministic=True):
        """
        Возвращает действие агента для наблюдения obs (numpy array).
        Args:
            obs: состояние среды (numpy array)
            deterministic: всегда True, TD3 не использует стохастический вывод
        Returns:
            action: numpy array
            None (для совместимости с Stable-Baselines3)
        """
        self.actor.eval()
        import torch
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else torch.FloatTensor(obs)
            action = self.actor(obs_tensor).cpu().numpy()
        return action.squeeze(), None
    
    def train(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Обучает TD3 агента на одном батче из буфера опыта.
        
        Args:
            batch_size: Размер батча для обучения
            
        Returns:
            Словарь с метриками обучения
        """
        self.total_it += 1
        
        # Выбираем случайный батч из буфера опыта
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
        
        # --- Обновление критика ---
        
        # Добавляем шум к следующему действию для сглаживания Q-значений
        with torch.no_grad():
            # Выбираем следующее действие согласно целевой политике
            next_action = self.actor_target(next_state)
            
            # Добавляем шум
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -self.max_action, self.max_action)
            
            # Вычисляем целевое Q-значение
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q
        
        # Вычисляем текущие Q-значения
        current_q1, current_q2 = self.critic(state, action)
        
        # Вычисляем функцию потерь критика
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Обновляем веса критика
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Обновление актора ---
        
        # Обновляем актора с задержкой
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Вычисляем функцию потерь актора
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            # Обновляем веса актора
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Мягкое обновление целевых сетей
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Возвращаем метрики обучения
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0
        }
    
    def save(self, filename: str):
        """
        Сохраняет модель в файл.
        
        Args:
            filename: Имя файла для сохранения
        """
        torch.save({
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it
        }, filename)
    
    def load(self, filename: str):
        """
        Загружает модель из файла.
        
        Args:
            filename: Имя файла для загрузки
        """
        checkpoint = torch.load(filename)
        
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.total_it = checkpoint["total_it"]
