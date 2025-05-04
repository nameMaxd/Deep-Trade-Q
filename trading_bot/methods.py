import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


from .utils import WINDOW_SIZE

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=WINDOW_SIZE, replay_freq=100):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    min_v = np.min(data)
    max_v = np.max(data)
    # state_size устанавливается в Agent.__init__, get_state принимает min_v,max_v
    state = get_state(data, 0, window_size, min_v=min_v, max_v=max_v)

    # Создаем tqdm прогресс-бар для шагов внутри эпизода
    progress_bar = tqdm(range(data_length), total=data_length, desc=f'Ep {episode+1}/{ep_count}', position=0, leave=True)
    
    buy_count = 0
    sell_count = 0
    hold_count = 0
    total_reward = 0
    
    for t in progress_bar:
        reward = 0
        next_state = get_state(data, t + 1, window_size, min_v=min_v, max_v=max_v)

        # select an action
        action = agent.act(state)

        # BUY
        if action == 1:
            # Проверяем, не превышен ли лимит позиций (максимум 5 позиций)
            if len(agent.inventory) < 5:  # Ограничиваем количество позиций
                agent.inventory.append(float(data[t][0]))
                reward = 2.0
                if len(agent.inventory) <= window_size:
                    reward += 1.0 * (window_size - len(agent.inventory))
                buy_count += 1
            else:
                # Если лимит позиций превышен, меняем действие на HOLD
                action = 0
                reward = -5.0  # Штраф за попытку превысить лимит

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = float(data[t][0]) - bought_price
                reward = delta * 200.0 + 10.0
                total_profit += delta
                if delta < 0:
                    reward -= 50.0
                sell_count += 1
            else:
                reward = -20.0
                # Если нечего продавать, меняем действие на HOLD
                action = 0

        # HOLD
        else:
            reward = -1.0 * (t / len(data)) * (len(agent.inventory) + 1)
            if agent.inventory:
                reward -= 0.1 * len(agent.inventory)
            hold_count += 1

        total_reward += reward
        done = (t == data_length - 1)
        
        # Обновляем прогресс-бар с текущей статистикой каждые 100 шагов
        if t % 100 == 0 or t == data_length - 1:
            progress_bar.set_postfix({
                'profit': f'{total_profit:.2f}',
                'reward': f'{total_reward:.2f}',
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count,
                'pos': len(agent.inventory)
            })
            
        agent.remember(state, action, reward, next_state, done)

        # replay every replay_freq steps to speed up training
        if len(agent.memory) > batch_size and t % replay_freq == 0:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # Закрываем прогресс-бар
    progress_bar.close()
    
    # Подробная статистика в конце эпизода
    logging.info(f"Epoch {episode+1}/{ep_count}: profit={total_profit:.2f}, avg_loss={np.mean(np.array(avg_loss)) if avg_loss else 'N/A'}")
    logging.info(f"Actions: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}, Final positions={len(agent.inventory)}")
    logging.info(f"Total reward: {total_reward:.2f}")

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug, min_v=None, max_v=None, return_deltas=False):
    import numpy as np
    # prepare evaluation: clear memory, deterministic policy
    agent.memory.clear()
    agent.epsilon = 0.0
    # compute normalization bounds
    if min_v is None or max_v is None:
        arr = np.array([d[0] if isinstance(d, (list, tuple, np.ndarray)) else d for d in data])
        min_v, max_v = np.min(arr), np.max(arr)
    # batch build states
    n = len(data) - 1
    states = np.vstack([get_state(data, t, window_size, min_v=min_v, max_v=max_v)[0] for t in range(n)])
    # batch predict q-values
    qvals = agent.model.predict(states, verbose=0)
    total_profit = 0
    history = []
    agent.inventory = []
    deltas = []
    # simulate
    for t, q in enumerate(qvals):
        # action selection via threshold
        if q[1] - q[0] > agent.buy_threshold:
            action = 1
        elif q[2] - q[0] > agent.buy_threshold:
            action = 2
        else:
            action = 0
        # BUY
        if action == 1:
            agent.inventory.append(data[t][0])
            history.append((data[t], "BUY"))
            if debug:
                logging.debug(f"Buy at: {format_currency(data[t][0] if isinstance(data[t], (list, tuple)) else data[t])}")
        # SELL
        elif action == 2 and agent.inventory:
            bought_price = agent.inventory.pop(0)
            delta = float(data[t][0]) - float(bought_price)
            total_profit += delta
            deltas.append(delta)
            history.append((data[t], "SELL"))
            if debug:
                logging.debug(f"Sell at: {format_currency(data[t])} | {format_position(delta)}")
        # HOLD
        else:
            history.append((data[t], "HOLD"))
    # liquidate remaining
    for buy_price in agent.inventory:
        delta = float(data[-1][0]) - float(buy_price)
        total_profit += delta
        deltas.append(delta)
    if return_deltas:
        return total_profit, deltas
    return total_profit, history
