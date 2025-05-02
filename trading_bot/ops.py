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


def get_state(data, t, window_size, min_v=None, max_v=None):
    """Возвращает state размерности (window_size+3,) = (window_size-1 сигмоид + SMA+EMA+RSI + vol_ratio), все признаки в [0,1]"""
    d = t - window_size + 1
    block = data[d: t + 1] if d >= 0 else (-d) * [data[0]] + data[0: t + 1]
    # fix: если длина блока < window_size, дублируем первый элемент
    if len(block) < window_size:
        block = [block[0]] * (window_size - len(block)) + block
    # separate price and volume
    prices = [p for p, v in block]
    volumes = [v for p, v in block]
    # price change sigmoids
    res = [sigmoid(prices[i + 1] - prices[i]) for i in range(window_size - 1)]
    import numpy as np
    import pandas as pd
    price_arr = np.array(prices)
    vol_arr = np.array(volumes)
    sma = np.mean(price_arr)
    ema = pd.Series(price_arr).ewm(span=window_size).mean().iloc[-1]
    delta = np.diff(price_arr)
    up = delta.clip(min=0)
    down = -delta.clip(max=0)
    roll_up = pd.Series(up).rolling(window_size-1).mean().iloc[-1] if len(up) >= window_size-1 else 0
    roll_down = pd.Series(down).rolling(window_size-1).mean().iloc[-1] if len(down) >= window_size-1 else 0
    rs = roll_up / roll_down if roll_down != 0 else 0
    rsi = 100 - 100 / (1 + rs) if roll_down != 0 else 100
    # нормализация SMA, EMA по min/max
    if min_v is None or max_v is None:
        min_v = np.min([p for p, v in data])
        max_v = np.max([p for p, v in data])
    sma_n = (sma - min_v) / (max_v - min_v) if max_v > min_v else 0
    ema_n = (ema - min_v) / (max_v - min_v) if max_v > min_v else 0
    rsi_n = rsi / 100.0
    # add volume ratio feature
    vol_ratio = vol_arr[-1] / (np.mean(vol_arr) + 1e-8)
    res += [sma_n, ema_n, rsi_n, vol_ratio]
    # Add momentum and volatility features
    mom = (price_arr[-1] - price_arr[0]) / price_arr[0] if price_arr[0] > 0 else 0
    mom_n = sigmoid(mom)
    vol_std = price_arr.std() / (max_v - min_v) if max_v > min_v else 0
    res += [mom_n, vol_std]
    # Логируем признаки только для первых 5 вызовов за запуск
    if not hasattr(get_state, '_log_count'):
        get_state._log_count = 0
    if get_state._log_count < 5:
        print(f'[get_state] t={t} block={block} sigmoids={res[:-6]} sma_n={sma_n:.4f} ema_n={ema_n:.4f} rsi_n={rsi_n:.4f} vol_ratio={vol_ratio:.4f} mom_n={mom_n:.4f} vol_std={vol_std:.4f}')
        get_state._log_count += 1
    return np.array([res])
