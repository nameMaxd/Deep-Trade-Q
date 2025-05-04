"""
Реализация TD3 агента согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
Часть 4: Функции для тренировки и оценки
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from tqdm import tqdm

# Импортируем классы из предыдущих файлов
# В реальном использовании нужно импортировать из других файлов
from td3_agent_part3 import TD3Agent

class TD3TrainingCallback(BaseCallback):
    """
    Колбэк для отслеживания процесса обучения TD3 агента.
    """
    def __init__(self, 
                 eval_env,
                 eval_freq: int = 1000,
                 n_eval_episodes: int = 5,
                 log_path: str = None,
                 verbose: int = 1):
        """
        Инициализация колбэка.
        
        Args:
            eval_env: Среда для оценки агента
            eval_freq: Частота оценки (в шагах)
            n_eval_episodes: Количество эпизодов для оценки
            log_path: Путь для сохранения логов
            verbose: Уровень подробности логов
        """
        super(TD3TrainingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        
        # Для отслеживания прогресса
        self.rewards = []
        self.portfolio_values = []
        self.drawdowns = []
        self.sharpe_ratios = []
        self.eval_steps = []
    
    def _on_step(self) -> bool:
        """
        Вызывается на каждом шаге обучения.
        
        Returns:
            True, если обучение должно продолжаться, False иначе
        """
        # Проверяем, нужно ли выполнять оценку
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                asset_info = getattr(self.eval_env, 'asset_names', None)
                if asset_info is not None:
                    print(f"[EVAL] Step {self.n_calls}: Evaluating on assets: {asset_info}")
                else:
                    print(f"[EVAL] Step {self.n_calls}: Evaluation started...")
            # Оцениваем агента
            mean_reward, portfolio_value, max_drawdown, sharpe_ratio = self._evaluate_agent()
            # Сохраняем результаты
            self.rewards.append(mean_reward)
            self.portfolio_values.append(portfolio_value)
            self.drawdowns.append(max_drawdown)
            self.sharpe_ratios.append(sharpe_ratio)
            self.eval_steps.append(self.n_calls)
            # Выводим информацию
            if self.verbose > 0:
                print(f"[EVAL] Step {self.n_calls}: Mean Reward: {mean_reward:.2f}, Portfolio Value: {portfolio_value:.2f}, Max Drawdown: {max_drawdown:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
            # Сохраняем модель, если это лучший результат
            if len(self.sharpe_ratios) > 1 and sharpe_ratio > max(self.sharpe_ratios[:-1]):
                if self.log_path is not None:
                    self.model.save(os.path.join(self.log_path, "best_model"))
                    if self.verbose > 0:
                        print(f"Saving best model with Sharpe Ratio: {sharpe_ratio:.2f}")
        return True
    
    def _evaluate_agent(self) -> Tuple[float, float, float, float]:
        """
        Оценивает агента на тестовой среде.
        
        Returns:
            Кортеж (средняя награда, стоимость портфеля, максимальная просадка, коэффициент Шарпа)
        """
        # Сохраняем награды и стоимости портфеля
        rewards = []
        portfolio_values = []
        
        # Выполняем несколько эпизодов
        for _ in range(self.n_eval_episodes):
            # Сбрасываем среду
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_portfolio_values = []
            
            # Выполняем эпизод
            while not done:
                # Выбираем действие
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Выполняем шаг в среде
                obs, reward, done, truncated, info = self.eval_env.step(action)
                done = done or truncated
                
                # Сохраняем награду и стоимость портфеля
                episode_reward += reward
                episode_portfolio_values.append(info['portfolio_value'])
            
            # Сохраняем результаты эпизода
            rewards.append(episode_reward)
            portfolio_values.append(episode_portfolio_values)
        
        # Вычисляем среднюю награду
        mean_reward = np.mean(rewards)
        
        # Вычисляем конечную стоимость портфеля (среднюю по всем эпизодам)
        final_portfolio_value = np.mean([values[-1] for values in portfolio_values])
        
        # Вычисляем максимальную просадку
        max_drawdown = 0
        for values in portfolio_values:
            # Преобразуем в numpy массив
            values_array = np.array(values)
            
            # Вычисляем максимальную просадку
            peak = np.maximum.accumulate(values_array)
            drawdown = (peak - values_array) / peak
            max_episode_drawdown = np.max(drawdown)
            
            # Обновляем максимальную просадку
            max_drawdown = max(max_drawdown, max_episode_drawdown)
        
        # Вычисляем коэффициент Шарпа
        # Для простоты используем доходность портфеля
        returns = []
        for values in portfolio_values:
            # Вычисляем доходность
            returns_array = np.diff(values) / values[:-1]
            returns.extend(returns_array)
        
        # Коэффициент Шарпа = (Средняя доходность - Безрисковая ставка) / Стандартное отклонение доходности
        # Для простоты считаем безрисковую ставку равной 0
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Годовой коэффициент
        
        return mean_reward, final_portfolio_value, max_drawdown, sharpe_ratio
    
    def plot_results(self):
        """
        Визуализирует результаты обучения.
        """
        if len(self.eval_steps) == 0:
            print("No evaluation data available.")
            return
        
        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График средней награды
        axes[0, 0].plot(self.eval_steps, self.rewards)
        axes[0, 0].set_title('Mean Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # График стоимости портфеля
        axes[0, 1].plot(self.eval_steps, self.portfolio_values)
        axes[0, 1].set_title('Portfolio Value')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)
        
        # График максимальной просадки
        axes[1, 0].plot(self.eval_steps, self.drawdowns)
        axes[1, 0].set_title('Max Drawdown')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True)
        
        # График коэффициента Шарпа
        axes[1, 1].plot(self.eval_steps, self.sharpe_ratios)
        axes[1, 1].set_title('Sharpe Ratio')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Сохраняем график, если указан путь
        if self.log_path is not None:
            plt.savefig(os.path.join(self.log_path, "training_results.png"))
        
        plt.show()

def train_td3_agent(
    env,
    eval_env,
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
    net_arch: List[int] = [256, 256, 128],
    batch_size: int = 256,
    start_timesteps: int = 10000,
    max_timesteps: int = 1000000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
    exploration_noise: float = 0.1,
    log_path: str = None,
    verbose: int = 1,
    report_freq: int = 1000
):
    """
    Обучает TD3 агента в заданной среде.
    
    Args:
        env: Среда для обучения
        eval_env: Среда для оценки
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
        batch_size: Размер батча для обучения
        start_timesteps: Количество шагов случайного исследования
        max_timesteps: Максимальное количество шагов обучения
        eval_freq: Частота оценки (в шагах)
        n_eval_episodes: Количество эпизодов для оценки
        exploration_noise: Шум для исследования
        log_path: Путь для сохранения логов
        verbose: Уровень подробности логов
        report_freq: Частота промежуточных отчетов
        
    Returns:
        Кортеж (обученный агент, колбэк с результатами)
    """
    # Создаем директорию для логов, если нужно
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
    
    # Инициализируем агента
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=discount,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        net_arch=net_arch
    )
    
    # Инициализируем колбэк
    callback = TD3TrainingCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=log_path,
        verbose=verbose
    )
    
    # Привязываем агента к колбэку
    callback.model = agent
    
    # Инициализируем переменные
    episode_rewards = []
    episode_profits = []
    episode_losses = []
    episode_sharpes = []
    num_trades = []
    
    state = env.reset()
    episode_reward = 0
    episode_profit = 0
    episode_loss = 0
    trades = 0
    sharpe = 0
    
    pbar = tqdm(range(1, max_timesteps+1), desc='TD3 Training', ascii=True, ncols=80)
    for t in pbar:
        episode_timesteps = 0
        
        # Выбираем действие
        if t < start_timesteps:
            # Случайное действие для исследования
            action = env.action_space.sample()
        else:
            # Действие от агента с шумом для исследования
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)
        
        # Выполняем шаг в среде
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        # Сохраняем переход в буфер опыта
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        agent.replay_buffer.add(state, action, next_state, reward, done)
        
        # Обновляем состояние и награду
        state = next_state
        episode_reward += reward
        
        # Обучаем агента, если набрали достаточно данных
        if t >= start_timesteps:
            agent.train(batch_size)
        
        # Вызываем колбэк
        callback.n_calls = t
        callback._on_step()
        
        # Если эпизод завершен
        if done:
            # Выводим информацию
            if verbose > 0 and episode_timesteps % 10 == 0:
                print(f"Episode {episode_timesteps}: {episode_timesteps} steps, reward = {episode_reward:.2f}")
            
            # Сбрасываем среду
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
        
        # Отчет каждые report_freq шагов
        if t % report_freq == 0 or t == 1:
            print(f"[TRAIN] Step {t}: Reward={episode_reward:.2f}, Portfolio={env.portfolio_value:.2f}, Sharpe={getattr(env, 'sharpe_ratio', 0):.2f}")
    
    # Закрываем прогресс-бар
    pbar.close()
    
    # Сохраняем финальную модель
    if log_path is not None:
        agent.save(os.path.join(log_path, "final_model"))
    
    return agent, callback
