"""
Реализация TD3 агента согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
Часть 5: Функции для оценки и визуализации результатов
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional
import torch
from tqdm import tqdm

def evaluate_td3_agent(
    agent,
    env,
    n_episodes: int = 10,
    render: bool = False,
    report_freq: int = 1000
) -> Dict[str, Any]:
    """
    Оценивает TD3 агента на заданной среде.
    
    Args:
        agent: TD3 агент для оценки
        env: Среда для оценки
        n_episodes: Количество эпизодов для оценки
        render: Визуализировать ли процесс
        report_freq: Частота промежуточных отчетов
        
    Returns:
        Словарь с результатами оценки
    """
    episode_rewards = []
    episode_profits = []
    episode_losses = []
    episode_sharpes = []
    num_trades = []
    
    for ep in tqdm(range(n_episodes), desc='OOS Evaluation', ncols=100):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_profit = 0
        ep_loss = 0
        trades = 0
        sharpe = 0
        t = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            # ... (логика подсчета профита, лосса, шарпа и сделок)
            t += 1
            if t % report_freq == 0:
                print(f"[OOS] Эпизод {ep+1}, шаг {t}: Reward={ep_reward:.2f}, Profit={ep_profit:.2f}, Loss={ep_loss:.2f}, Sharpe={sharpe:.2f}, Trades={trades}")
            state = next_state
        episode_rewards.append(ep_reward)
        episode_profits.append(ep_profit)
        episode_losses.append(ep_loss)
        episode_sharpes.append(sharpe)
        num_trades.append(trades)
    
    # Сохраняем результаты
    all_rewards = episode_rewards
    all_portfolio_values = []
    all_returns = []
    all_drawdowns = []
    all_trades = []
    
    # Выполняем несколько эпизодов
    for ep in range(n_episodes):
        # Сбрасываем среду
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_portfolio_values = []
        episode_trades = {"buy": 0, "sell": 0, "hold": 0}
        
        # Выполняем эпизод
        while not done:
            # Выбираем действие
            action = agent.select_action(obs)
            
            # Выполняем шаг в среде
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Сохраняем награду и стоимость портфеля
            episode_reward += reward
            episode_portfolio_values.append(info['portfolio_value'])
            
            # Отслеживаем сделки
            if 'buy_trades' in info and info['buy_trades'] > episode_trades["buy"]:
                episode_trades["buy"] += 1
            if 'sell_trades' in info and info['sell_trades'] > episode_trades["sell"]:
                episode_trades["sell"] += 1
            
            # Визуализируем, если нужно
            if render:
                env.render()
            
            # Обновляем состояние
            obs = next_obs
        
        # Вычисляем доходность
        episode_returns = np.diff(episode_portfolio_values) / episode_portfolio_values[:-1]
        
        # Вычисляем максимальную просадку
        peak = np.maximum.accumulate(episode_portfolio_values)
        drawdown = (peak - episode_portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Сохраняем результаты эпизода
        all_rewards.append(episode_reward)
        all_portfolio_values.append(episode_portfolio_values)
        all_returns.extend(episode_returns)
        all_drawdowns.append(max_drawdown)
        all_trades.append(episode_trades)
        
        # Выводим информацию
        if True:
            print(f"Episode {ep+1}/{n_episodes}: Reward = {episode_reward:.2f}, "
                  f"Final Value = {episode_portfolio_values[-1]:.2f}, "
                  f"Max Drawdown = {max_drawdown:.2%}, "
                  f"Trades: Buy = {episode_trades['buy']}, Sell = {episode_trades['sell']}")
    
    # Вычисляем итоговые метрики
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_final_value = np.mean([values[-1] for values in all_portfolio_values])
    mean_max_drawdown = np.mean(all_drawdowns)
    
    # Вычисляем коэффициент Шарпа
    sharpe_ratio = np.mean(all_returns) / (np.std(all_returns) + 1e-6) * np.sqrt(252)  # Годовой коэффициент
    
    # Вычисляем среднее количество сделок
    mean_buy_trades = np.mean([trades["buy"] for trades in all_trades])
    mean_sell_trades = np.mean([trades["sell"] for trades in all_trades])
    
    # Формируем результат
    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_final_value": mean_final_value,
        "mean_max_drawdown": mean_max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "mean_buy_trades": mean_buy_trades,
        "mean_sell_trades": mean_sell_trades,
        "all_portfolio_values": all_portfolio_values
    }
    
    return results

def visualize_trading_results(
    results: Dict[str, Any],
    price_data: pd.DataFrame,
    save_path: str = None,
    show: bool = True
):
    """
    Визуализирует результаты торговли.
    
    Args:
        results: Словарь с результатами оценки
        price_data: DataFrame с ценами
        save_path: Путь для сохранения графика
        show: Показывать ли график
    """
    # Создаем графики
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    # График цены
    axes[0].plot(price_data['Close'].values, label='Close Price', color='black')
    axes[0].set_title('Price Chart')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # График стоимости портфеля
    for i, values in enumerate(results["all_portfolio_values"]):
        axes[1].plot(values, label=f'Episode {i+1}', alpha=0.7)
    
    axes[1].set_title('Portfolio Value')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # График доходности
    returns = []
    for values in results["all_portfolio_values"]:
        returns.append(np.diff(values) / values[:-1])
    
    for i, ret in enumerate(returns):
        axes[2].plot(ret, label=f'Episode {i+1}', alpha=0.7)
    
    axes[2].set_title('Returns')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Return')
    axes[2].legend()
    axes[2].grid(True)
    
    # Добавляем информацию о метриках
    plt.figtext(0.01, 0.01, 
                f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n"
                f"Mean Final Value: {results['mean_final_value']:.2f}\n"
                f"Mean Max Drawdown: {results['mean_max_drawdown']:.2%}\n"
                f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"Mean Buy Trades: {results['mean_buy_trades']:.1f}\n"
                f"Mean Sell Trades: {results['mean_sell_trades']:.1f}",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    # Сохраняем график, если указан путь
    if save_path is not None:
        plt.savefig(save_path)
    
    # Показываем график, если нужно
    if show:
        plt.show()
    else:
        plt.close()

def backtest_td3_agent(
    agent,
    price_data: pd.DataFrame,
    window_size: int = 47,
    initial_balance: float = 10000.0,
    commission: float = 0.001,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Выполняет бэктест TD3 агента на исторических данных.
    
    Args:
        agent: TD3 агент для бэктеста
        price_data: DataFrame с ценами
        window_size: Размер окна для анализа
        initial_balance: Начальный баланс
        commission: Комиссия за сделку
        verbose: Уровень подробности логов
        
    Returns:
        Словарь с результатами бэктеста
    """
    # Инициализируем переменные
    balance = initial_balance
    inventory = 0
    avg_buy_price = 0
    portfolio_values = [initial_balance]
    actions = []
    positions = []
    
    # Создаем прогресс-бар
    pbar = tqdm(total=len(price_data) - window_size, desc="Backtesting")
    
    # Выполняем бэктест
    for t in range(len(price_data) - window_size):
        # Получаем текущие данные
        current_data = price_data.iloc[t:t+window_size]
        current_price = current_data['Close'].iloc[-1]
        
        # Формируем состояние
        # Здесь нужно реализовать формирование состояния так же, как в среде
        # Для простоты используем заглушку
        state = np.zeros(14)  # 1 + 13 * 1 (для одного актива)
        state[0] = balance / initial_balance
        state[1] = current_price / current_data['Close'].mean()
        state[2] = inventory / 8  # Максимальное количество позиций
        
        # Выбираем действие
        action = agent.select_action(state)
        actions.append(action[0])
        
        # Интерпретируем действие
        if action[0] > 0:  # Покупка
            # Максимальное количество акций, которое можно купить
            max_buy = min(
                balance / (current_price * (1 + commission)) / 8,
                8 - inventory
            )
            shares_to_buy = max(0, action[0] * max_buy)
            
            # Проверяем минимальный размер сделки
            if shares_to_buy * current_price < 1000:
                shares_to_buy = 0
                
            # Выполняем покупку
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + commission)
                balance -= cost
                
                # Обновляем среднюю цену покупки
                if inventory == 0:
                    avg_buy_price = current_price
                else:
                    avg_buy_price = (avg_buy_price * inventory + current_price * shares_to_buy) / (inventory + shares_to_buy)
                
                inventory += shares_to_buy
                
        elif action[0] < 0:  # Продажа
            shares_to_sell = min(inventory, -action[0] * inventory)
            
            # Проверяем минимальный размер сделки
            if shares_to_sell * current_price < 1000:
                shares_to_sell = 0
                
            # Выполняем продажу
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - commission)
                balance += revenue
                inventory -= shares_to_sell
                
                # Если продали всё, сбрасываем среднюю цену покупки
                if inventory == 0:
                    avg_buy_price = 0
        
        # Рассчитываем стоимость портфеля
        portfolio_value = balance + (inventory * current_price)
        portfolio_values.append(portfolio_value)
        positions.append(inventory)
        
        # Обновляем прогресс-бар
        pbar.update(1)
    
    # Закрываем прогресс-бар
    pbar.close()
    
    # Вычисляем метрики
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    # Вычисляем максимальную просадку
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    # Вычисляем количество сделок
    buy_trades = 0
    sell_trades = 0
    for i in range(1, len(positions)):
        if positions[i] > positions[i-1]:
            buy_trades += 1
        elif positions[i] < positions[i-1]:
            sell_trades += 1
    
    # Формируем результат
    results = {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "final_inventory": inventory,
        "final_portfolio_value": portfolio_values[-1],
        "return": (portfolio_values[-1] - initial_balance) / initial_balance,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "portfolio_values": portfolio_values,
        "actions": actions,
        "positions": positions
    }
    
    # Выводим информацию
    if verbose > 0:
        print(f"Initial Balance: {initial_balance:.2f}")
        print(f"Final Balance: {balance:.2f}")
        print(f"Final Inventory: {inventory:.2f} shares")
        print(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
        print(f"Return: {results['return']:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Buy Trades: {buy_trades}")
        print(f"Sell Trades: {sell_trades}")
    
    return results
