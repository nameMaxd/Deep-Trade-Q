import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, window_size):
    """Возвращает state размерности (window_size+2,) = (window_size-1 сигмоид + SMA+EMA+RSI)"""
    d = t - window_size + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]
    res = [sigmoid(block[i + 1] - block[i]) for i in range(window_size - 1)]
    import numpy as np
    import pandas as pd
    block_arr = np.array(block)
    sma = np.mean(block_arr)
    ema = pd.Series(block_arr).ewm(span=window_size).mean().iloc[-1]
    delta = np.diff(block_arr)
    up = delta.clip(min=0)
    down = -delta.clip(max=0)
    roll_up = pd.Series(up).rolling(window_size-1).mean().iloc[-1] if len(up) >= window_size-1 else 0
    roll_down = pd.Series(down).rolling(window_size-1).mean().iloc[-1] if len(down) >= window_size-1 else 0
    rs = roll_up / roll_down if roll_down != 0 else 0
    rsi = 100 - 100 / (1 + rs) if roll_down != 0 else 100
    res += [sma, ema, rsi]
    # Логируем признаки только для первых 5 вызовов за запуск
    if not hasattr(get_state, '_log_count'):
        get_state._log_count = 0
    if get_state._log_count < 5:
        print(f'[get_state] t={t} block={block} sigmoids={res[:-3]} sma={sma:.4f} ema={ema:.4f} rsi={rsi:.4f}')
        get_state._log_count += 1
    return np.array([res])
