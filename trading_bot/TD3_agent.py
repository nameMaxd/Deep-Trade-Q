"""
TD3 Agent для Deep-Trade-Q
Содержит классы и функции для обучения и инференса TD3 агента.
"""

import os
import logging
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from config import *  # Импортируем параметры из конфига


# Вспомогательные функции для метрик
def max_drawdown(returns):
    """Вычисляет максимальную просадку из списка доходностей"""
    if not returns:
        return 0.0
    cumulative = np.maximum.accumulate(returns)
    drawdowns = cumulative - returns
    return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

def cvar(returns, alpha=0.05):
    """Вычисляет Conditional Value at Risk (CVaR) для заданного уровня alpha"""
    if not returns or len(returns) < 2:
        return 0.0
    returns = np.array(returns)
    n = int(np.ceil(alpha * len(returns)))
    if n == 0:
        n = 1
    sorted_returns = np.sort(returns)
    return -np.mean(sorted_returns[:n]) if n > 0 else 0.0


# Progress bar для TD3
class TqdmCallback(BaseCallback):
    """Callback для отображения прогресса обучения TD3 с tqdm"""
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc='TD3 Training')
        
    def _on_step(self) -> bool:
        # update progress
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True
        
    def _on_training_end(self):
        self.pbar.close()


# Callback для оценки и раннего останова
class EvalCallbackTD3(BaseCallback):
    """Callback для оценки модели и раннего останова"""
    def __init__(self, train_env, eval_env, eval_freq, n_eval_episodes, patience, save_path):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.best_val_profit = -float('inf')
        self.no_improve = 0
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            # Train evaluation (with noise)
            tp, tr, ttrades = [], [], []
            for i in range(self.n_eval_episodes):
                obs, _ = self.train_env.reset(random_start=True); done=False; p_t=0.0; r_t=0.0; trade_t=0; steps=0
                buy_t = sell_t = hold_t = 0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, rew, done, _, info = self.train_env.step(act)
                    real_action = info.get('real_action', act)
                    if real_action == 0:
                        hold_t += 1
                    elif real_action == 1:
                        buy_t += 1; trade_t += 1
                    elif real_action == 2:
                        # Count only successful sells
                        if self.train_env.inventory:
                            sell_t += 1; trade_t += 1
                        else:
                            hold_t += 1
                    if real_action == 2:
                        p_t += rew
                    r_t += rew; steps += 1
                # liquidate remaining positions at end of episode (handle price,qty)
                if getattr(self.train_env, 'inventory', None):
                    final_price = float(self.train_env.prices[self.train_env.current_step])
                    for bought_price, qty in self.train_env.inventory:
                        profit = (final_price - bought_price) * qty
                        cost = self.train_env.commission * (final_price * qty + bought_price * qty)
                        p_t += profit - cost
                    self.train_env.inventory.clear()
                tp.append(p_t); tr.append(r_t/steps if steps else 0.0); ttrades.append(trade_t)
                logging.info(f"[Eval] train ep {i}: profit {p_t:.6f}, buys {buy_t}, sells {sell_t}, holds {hold_t}")
            
            # Validation evaluation (with noise)
            vp, vr, vtrades = [], [], []
            for i in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset(random_start=True); done=False; p_v=0.0; r_v=0.0; trade_v=0; steps=0
                buy_v = sell_v = hold_v = 0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, rew, done, _, info = self.eval_env.step(act)
                    real_action = info.get('real_action', act)
                    if real_action == 0:
                        hold_v += 1
                    elif real_action == 1:
                        buy_v += 1; trade_v += 1
                    elif real_action == 2:
                        if self.eval_env.inventory:
                            sell_v += 1; trade_v += 1
                        else:
                            hold_v += 1
                    if real_action == 2:
                        p_v += rew
                    r_v += rew; steps += 1
                # liquidate remaining positions at end of episode (handle price,qty)
                if getattr(self.eval_env, 'inventory', None):
                    final_price = float(self.eval_env.prices[self.eval_env.current_step])
                    for bought_price, qty in self.eval_env.inventory:
                        profit = (final_price - bought_price) * qty
                        cost = self.eval_env.commission * (final_price * qty + bought_price * qty)
                        p_v += profit - cost
                    self.eval_env.inventory.clear()
                vp.append(p_v); vr.append(r_v/steps if steps else 0.0); vtrades.append(trade_v)
                logging.info(f"[Eval] val ep {i}: profit {p_v:.6f}, buys {buy_v}, sells {sell_v}, holds {hold_v}")
            
            # Print individual episode profits for clarity
            best_train, worst_train = max(tp), min(tp)
            best_val, worst_val = max(vp), min(vp)
            
            # Also log for file
            for i, p in enumerate(tp): logging.info(f"[Eval] train ep {i}:profit {p:.6f}")
            for i, p in enumerate(vp): logging.info(f"[Eval] val ep {i}:profit {p:.6f}")
            
            mpt, mrt, mtt = np.mean(tp), np.mean(tr), np.mean(ttrades)
            mpv, mrv, mv = np.mean(vp), np.mean(vr), np.mean(vtrades)
            
            # === Новые метрики: Sharpe, max drawdown, CVaR, Omega, композитный score ===
            from trading_bot.utils import omega
            if len(vp) > 1:
                std_vp = np.std(vp, ddof=1)
                sharpe = mpv / std_vp if std_vp > 0 else 0.0
            else:
                sharpe = 0.0
                
            train_max_dd = max_drawdown(tp)
            val_max_dd = max_drawdown(vp)
            train_cvar = cvar(tp)
            val_cvar = cvar(vp)
            train_omega = omega(tp)
            val_omega = omega(vp)
            
            alpha, beta, gamma, delta = 0.5, 0.2, 0.2, 0.2
            val_score = mpv - alpha*std_vp - beta*abs(worst_val) - gamma*val_max_dd - delta*val_cvar
            train_score = mpt - alpha*np.std(tp, ddof=1) - beta*abs(worst_train) - gamma*train_max_dd - delta*train_cvar
            
            # Логируем результаты
            logging.info(f"[Eval] Step {self.num_timesteps}: TrainProfit {mpt:.6f}, ValProfit {mpv:.6f}, Sharpe {sharpe:.6f}, AvgReward {mrv:.6f}, TradesTrain {mtt}, TradesVal {mv}")
            logging.info(f"[Eval] MaxDD train {train_max_dd:.6f}, val {val_max_dd:.6f}; CVaR train {train_cvar:.6f}, val {val_cvar:.6f}; Omega train {train_omega:.6f}, val {val_omega:.6f}")
            
            # early stopping по профиту
            if mpv > self.best_val_profit:
                self.best_val_profit = mpv
                self.no_improve = 0
                self.model.save(self.save_path)
                logging.info(f"[Eval] Model improved and saved to {self.save_path}")
            else:
                self.no_improve += 1
                logging.info(f"[Eval] No improvement for {self.no_improve} evals")
                
            if self.no_improve >= self.patience:
                logging.info(f"[Eval] Early stopping triggered at step {self.num_timesteps}")
                return False
                
            return True


# Callback для визуализации
class VisualizeCallback(BaseCallback):
    """Callback для визуализации процесса обучения"""
    def __init__(self, train_env, val_env, model, train_env_raw, val_env_raw, total_timesteps):
        super().__init__()
        self.train_env = train_env
        self.val_env = val_env
        self.model = model
        self.train_env_raw = train_env_raw
        self.val_env_raw = val_env_raw
        self.total_timesteps = total_timesteps
        self.best_val_profit = -float('inf')
        self.no_improve = 0
        self.patience = 5  # Дефолтное значение
        self.save_path = None

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            # Train evaluation (with noise)
            tp, tr, ttrades = [], [], []
            for i in range(3):
                obs, _ = self.train_env.reset(random_start=True); done=False; p_t=0.0; r_t=0.0; trade_t=0; steps=0
                buy_t = sell_t = hold_t = 0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, rew, done, _, info = self.train_env.step(act)
                    real_action = info.get('real_action', act)
                    if real_action == 0:
                        hold_t += 1
                    elif real_action == 1:
                        buy_t += 1; trade_t += 1
                    elif real_action == 2:
                        # Count only successful sells
                        if self.train_env.inventory:
                            sell_t += 1; trade_t += 1
                        else:
                            hold_t += 1
                    if real_action == 2:
                        p_t += rew
                    r_t += rew; steps += 1
                # liquidate remaining positions at end of episode (handle price,qty)
                if getattr(self.train_env, 'inventory', None):
                    final_price = float(self.train_env.prices[self.train_env.current_step])
                    for bought_price, qty in self.train_env.inventory:
                        profit = (final_price - bought_price) * qty
                        cost = self.train_env.commission * (final_price * qty + bought_price * qty)
                        p_t += profit - cost
                    self.train_env.inventory.clear()
                tp.append(p_t); tr.append(r_t/steps if steps else 0.0); ttrades.append(trade_t)
                logging.info(f"[Eval] train ep {i}: profit {p_t:.6f}, buys {buy_t}, sells {sell_t}, holds {hold_t}")
            
            # Validation evaluation (with noise)
            vp, vr, vtrades = [], [], []
            for i in range(3):
                obs, _ = self.val_env.reset(random_start=True); done=False; p_v=0.0; r_v=0.0; trade_v=0; steps=0
                buy_v = sell_v = hold_v = 0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, rew, done, _, info = self.val_env.step(act)
                    real_action = info.get('real_action', act)
                    if real_action == 0:
                        hold_v += 1
                    elif real_action == 1:
                        buy_v += 1; trade_v += 1
                    elif real_action == 2:
                        if self.val_env.inventory:
                            sell_v += 1; trade_v += 1
                        else:
                            hold_v += 1
                    if real_action == 2:
                        p_v += rew
                    r_v += rew; steps += 1
                # liquidate remaining positions at end of episode (handle price,qty)
                if getattr(self.val_env, 'inventory', None):
                    final_price = float(self.val_env.prices[self.val_env.current_step])
                    for bought_price, qty in self.val_env.inventory:
                        profit = (final_price - bought_price) * qty
                        cost = self.val_env.commission * (final_price * qty + bought_price * qty)
                        p_v += profit - cost
                    self.val_env.inventory.clear()
                vp.append(p_v); vr.append(r_v/steps if steps else 0.0); vtrades.append(trade_v)
                logging.info(f"[Eval] val ep {i}: profit {p_v:.6f}, buys {buy_v}, sells {sell_v}, holds {hold_v}")
            
            # Print individual episode profits for clarity
            best_train, worst_train = max(tp), min(tp)
            best_val, worst_val = max(vp), min(vp)
            
            # Also log for file
            for i, p in enumerate(tp): logging.info(f"[Eval] train ep {i}:profit {p:.6f}")
            for i, p in enumerate(vp): logging.info(f"[Eval] val ep {i}:profit {p:.6f}")
            
            mpt, mrt, mtt = np.mean(tp), np.mean(tr), np.mean(ttrades)
            mpv, mrv, mv = np.mean(vp), np.mean(vr), np.mean(vtrades)
            
            # === Новые метрики: Sharpe, max drawdown, CVaR, Omega, композитный score ===
            from trading_bot.utils import omega
            if len(vp) > 1:
                std_vp = np.std(vp, ddof=1)
                sharpe = mpv / std_vp if std_vp > 0 else 0.0
            else:
                sharpe = 0.0
                
            train_max_dd = max_drawdown(tp)
            val_max_dd = max_drawdown(vp)
            train_cvar = cvar(tp)
            val_cvar = cvar(vp)
            train_omega = omega(tp)
            val_omega = omega(vp)
            
            alpha, beta, gamma, delta = 0.5, 0.2, 0.2, 0.2
            val_score = mpv - alpha*std_vp - beta*abs(worst_val) - gamma*val_max_dd - delta*val_cvar
            train_score = mpt - alpha*np.std(tp, ddof=1) - beta*abs(worst_train) - gamma*train_max_dd - delta*train_cvar
            
            # Логируем результаты
            logging.info(f"[Eval] Step {self.num_timesteps}: TrainProfit {mpt:.6f}, ValProfit {mpv:.6f}, Sharpe {sharpe:.6f}, AvgReward {mrv:.6f}, TradesTrain {mtt}, TradesVal {mv}")
            logging.info(f"[Eval] MaxDD train {train_max_dd:.6f}, val {val_max_dd:.6f}; CVaR train {train_cvar:.6f}, val {val_cvar:.6f}; Omega train {train_omega:.6f}, val {val_omega:.6f}")
            
            # Если есть save_path, сохраняем лучшую модель
            if self.save_path is not None:
                if mpv > self.best_val_profit:
                    self.best_val_profit = mpv
                    self.no_improve = 0
                    self.model.save(self.save_path)
                    logging.info(f"[Eval] Model improved and saved to {self.save_path}")
                else:
                    self.no_improve += 1
                    logging.info(f"[Eval] No improvement for {self.no_improve} evals")
                    
                if self.no_improve >= self.patience:
                    logging.info(f"[Eval] Early stopping triggered at step {self.num_timesteps}")
                    return False
                    
            return True


def create_TD3_model(env, td3_noise_sigma, tb_dir):
    """Создает и возвращает модель TD3 с параметрами из конфига"""
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=td3_noise_sigma * np.ones(n_actions))
    
    # Initialize TD3 с параметрами для стабильного обучения и предотвращения переобучения
    model = TD3(
        'MlpPolicy', 
        env, 
        policy_kwargs={
            'activation_fn': torch.nn.ReLU,  # Используем ReLU для лучшего обучения
            'net_arch': {
                'pi': [128, 64, 32],  # Более широкая архитектура политики (actor)
                'qf': [256, 128, 64]  # Более широкая архитектура Q-функции (critic)
            }
        },
        action_noise=action_noise, 
        verbose=TD3_VERBOSE,
        tensorboard_log=tb_dir,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=256,       # Увеличиваем размер батча для стабильности
        buffer_size=50000,    # Увеличиваем буфер для лучшего обучения
        train_freq=TD3_TRAIN_FREQ,
        gradient_steps=TD3_GRADIENT_STEPS,
        learning_starts=TD3_LEARNING_STARTS,
        tau=TD3_TAU,
        policy_delay=TD3_POLICY_DELAY,
        target_policy_noise=TD3_POLICY_NOISE,
        target_noise_clip=TD3_NOISE_CLIP,
        seed=TD3_SEED
    )
    
    return model


def train_TD3(model, train_env, val_env, train_env_raw, val_env_raw, td3_timesteps, td3_save_name, stock, monitor_dir, plots_dir):
    """Обучает модель TD3 и сохраняет результаты"""
    # Передаем оригинальные окружения для визуализации
    visual_cb = VisualizeCallback(train_env, val_env, model, train_env_raw, val_env_raw, total_timesteps=td3_timesteps)
    
    # Запускаем обучение в несколько эпизодов
    num_episodes = 10  # Увеличиваем с 5 до 10 эпизодов
    timesteps_per_episode = td3_timesteps // num_episodes
    
    for episode in trange(num_episodes, desc='Episodes', position=0, leave=True):
        logging.info(f'Start episode {episode + 1}/{num_episodes}')
        model.learn(total_timesteps=timesteps_per_episode, callback=visual_cb, reset_num_timesteps=False)
        logging.info(f'End episode {episode + 1}/{num_episodes}')
    
    model.save(f"{td3_save_name}")
    
    # Сохраняем график наград из логов Monitor
    try:
        # Пытаемся прочитать логи из Monitor
        monitor_file = os.path.join(monitor_dir, 'monitor.csv')
        if os.path.exists(monitor_file):
            # Читаем CSV с наградами
            df = pd.read_csv(monitor_file, skiprows=1)  # Пропускаем строку с комментарием
            
            if 'r' in df.columns and len(df) > 0:
                # Преобразуем данные в numpy массивы для избежания ошибки с индексированием
                x = np.arange(len(df))
                y = df['r'].values  # Используем .values для получения numpy массива
                
                plt.figure(figsize=(12, 4))
                plt.plot(x, y, label='Episode Reward')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'TD3 Training Rewards - {stock}')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, 'rewards.png'))
                plt.close()
    except Exception as e:
        logging.error(f"Ошибка при построении графика наград: {e}")
    
    return model


def train_TD3_with_callbacks(model, train_env, val_env, train_eval_env, td3_timesteps, td3_save_name, stock):
    """Обучает модель TD3 с использованием callbacks для оценки и раннего останова"""
    save_path = f'{td3_save_name}_{os.path.splitext(stock)[0]}'
    tqdm_cb = TqdmCallback(td3_timesteps)
    eval_cb = EvalCallbackTD3(
        train_eval_env,
        val_env,
        eval_freq=1000,
        n_eval_episodes=3,
        patience=5,
        save_path=save_path
    )
    
    # Callbacks: progress bar + eval
    callbacks = [tqdm_cb, eval_cb]
    
    # Запускаем обучение
    model.learn(total_timesteps=td3_timesteps, callback=callbacks)
    
    # Сохраняем модель
    model.save(f"{td3_save_name}")
    
    return model, save_path
