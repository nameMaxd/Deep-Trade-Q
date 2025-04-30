import os
import math
import logging

import pandas as pd
import numpy as np

from tensorflow.keras import backend as K


# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def show_eval_result(model_name, profit, initial_offset):
    """ Displays eval results
    """
    # If profit is a tuple, use the first element
    if isinstance(profit, tuple):
        profit_val = profit[0]
    else:
        profit_val = profit
    if profit_val == initial_offset or profit_val == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit_val)))


WINDOW_SIZE = 50  # Fixed window size for all modules

def minmax_normalize(arr):
    arr = np.array(arr)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if max_v - min_v == 0:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)

def zscore_normalize(arr):
    arr = np.array(arr)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std

def log_returns(arr):
    arr = np.array(arr)
    arr = np.where(arr <= 0, np.nan, arr)
    log_ret = np.diff(np.log(arr))
    log_ret = np.insert(log_ret, 0, 0)
    return log_ret

def fillna_inf(arr):
    arr = np.array(arr)
    mask = np.isnan(arr) | np.isinf(arr)
    if np.any(mask):
        arr[mask] = np.nanmean(arr[~mask]) if np.any(~mask) else 0
    return arr

def get_stock_data(stock_file, norm_type="minmax"):
    """Reads stock data from csv file. norm_type: minmax, zscore, log-returns. Заполняет NaN/inf скользящим средним."""
    import os
    if os.path.isfile(stock_file):
        path = stock_file
    else:
        path = os.path.join('data', f'{stock_file}.csv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File '{stock_file}' or '{path}' not found.")
    df = pd.read_csv(path)
    prices = np.array(df['Adj Close'])
    prices = fillna_inf(prices)
    if norm_type == "zscore":
        prices = zscore_normalize(prices)
    elif norm_type == "log-returns":
        prices = log_returns(prices)
    else:
        prices = minmax_normalize(prices)
    return list(prices)


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
