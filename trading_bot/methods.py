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

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=WINDOW_SIZE):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    min_v = np.min(data)
    max_v = np.max(data)
    state = get_state(data, 0, window_size, min_v=min_v, max_v=max_v)

    # tqdm будет печатать прогресс только в консоль, не в лог
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count), file=None):
        reward = 0
        next_state = get_state(data, t + 1, window_size, min_v=min_v, max_v=max_v)

        # select an action
        action = agent.act(state)
        # Логируем action, reward, inventory (только первые 20 шагов)
        if t < 20:
            print(f'[train_model] t={t} action={action} reward={reward:.4f} inventory={agent.inventory}')

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            # штраф за переполнение inventory (если держим слишком долго)
            if len(agent.inventory) > WINDOW_SIZE:
                reward -= 0.05 * (len(agent.inventory) - WINDOW_SIZE)

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = data[t] - bought_price
                if delta > 0:
                    reward = delta * 1.0
                else:
                    reward = -abs(delta) * 1.0
                total_profit += delta
            else:
                reward = -0.05  # штраф за SELL без inventory
                print(f'[train_model] t={t} action=SELL без inventory -> штраф')

        # HOLD
        else:
            # поощрение за удержание при росте
            if len(agent.inventory) > 0:
                last_buy = agent.inventory[0]
                reward += 0.01 * (data[t] - last_buy) / (last_buy + 1e-8)
            # базовый штраф за пассивность
            reward -= 0.005

        done = (t == data_length - 1)
        # Логируем reward (только первые 10 шагов)
        if t < 10:
            print(f'[train_model] t={t} reward={reward:.4f}')
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # НЕ сохраняем веса внутри train_model при по-недельном обучении!
    # if episode % 10 == 0:
    #     agent.save(episode)

    # Итоги эпохи логируем кратко (номер, профит, средний лосс)
    logging.info(f"Epoch {episode}/{ep_count}: profit={total_profit:.2f}, avg_loss={np.mean(np.array(avg_loss)) if avg_loss else 'N/A'}")

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    
    state = get_state(data, 0, window_size)

    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, window_size, min_v=min_v, max_v=max_v)
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
