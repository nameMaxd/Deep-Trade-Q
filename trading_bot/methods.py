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

    # tqdm будет печатать прогресс только в консоль, не в лог
    for t in tqdm(range(data_length), total=data_length, desc=f'Ep {episode}/{ep_count}', dynamic_ncols=True, leave=True):
        reward = 0
        next_state = get_state(data, t + 1, window_size, min_v=min_v, max_v=max_v)

        # select an action
        action = agent.act(state)
        # Логируем action, reward, inventory (только первые 20 шагов)
        if t < 20:
            print(f'[train_model] t={t} action={action} reward={reward:.4f} inventory={agent.inventory}')

        # BUY
        if action == 1:
            agent.inventory.append(float(data[t][0]))
            # Штраф за превышение окна удержания
            if len(agent.inventory) > WINDOW_SIZE:
                reward -= 0.1 * (len(agent.inventory) - WINDOW_SIZE)
            else:
                reward += 0.5  # Бонус за открытие позиции

        # SELL
        elif action == 2:
            if len(agent.inventory) > 0:
                bought_price = float(agent.inventory.pop(0))
                curr_price = float(data[t][0])
                delta = curr_price - bought_price
                reward = delta * 100.0  # Увеличиваем масштаб награды
                total_profit += delta
            else:
                reward = -1.0  # Жесткий штраф за попытку продажи без позиции

        # HOLD
        else:
            reward = -0.5  # Большой штраф за бездействие
            if len(agent.inventory) > 0:
                last_buy = float(agent.inventory[0])
                curr_price = float(data[t][0])
                reward += 0.01 * (curr_price - last_buy) / (last_buy + 1e-8)
                # штраф за слишком долгие позиции (каждая позиция)
                reward -= 0.005 * len(agent.inventory)

        done = (t == data_length - 1)
        # Логируем reward (только первые 10 шагов)
        if t < 10:
            print(f'[train_model] t={t} reward={reward:.4f}')
        agent.remember(state, action, reward, next_state, done)

        # replay every replay_freq steps to speed up training
        if len(agent.memory) > batch_size and t % replay_freq == 0:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    # НЕ сохраняем веса внутри train_model при по-недельном обучении!
    # if episode % 10 == 0:
    #     agent.save(episode)

    # Итоги эпохи логируем кратко (номер, профит, средний лосс)
    logging.info(f"Epoch {episode}/{ep_count}: profit={total_profit:.2f}, avg_loss={np.mean(np.array(avg_loss)) if avg_loss else 'N/A'}")

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug, min_v=None, max_v=None):
    import numpy as np
    # prepare evaluation: clear memory, deterministic policy
    agent.memory.clear()
    agent.epsilon = 0.0
    # compute normalization bounds
    if min_v is None or max_v is None:
        arr = np.array([d[0] for d in data])
        min_v, max_v = np.min(arr), np.max(arr)
    # batch build states
    n = len(data) - 1
    states = np.vstack([get_state(data, t, window_size, min_v=min_v, max_v=max_v)[0] for t in range(n)])
    # batch predict q-values
    qvals = agent.model.predict(states, verbose=0)
    total_profit = 0
    history = []
    agent.inventory = []
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
                logging.debug(f"Buy at: {format_currency(data[t])}")
        # SELL
        elif action == 2 and agent.inventory:
            bought_price = agent.inventory.pop(0)
            delta = float(data[t][0]) - float(bought_price)
            total_profit += delta
            history.append((data[t], "SELL"))
            if debug:
                logging.debug(f"Sell at: {format_currency(data[t])} | {format_position(delta)}")
        # HOLD
        else:
            history.append((data[t], "HOLD"))
    # liquidate remaining
    for buy_price in agent.inventory:
        total_profit += float(data[-1][0]) - float(buy_price)
    return total_profit, history
