"""
Script for training Stock Trading Bot.

Usage:
  train.py <stock> [<dummy>] [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] [--model-type=<model-type>]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 20]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
  --model-type=<model-type>         Model type: 'dense' (default) or 'lstm'.

"""

import logging
import coloredlogs
import os
import tensorflow as tf
import io
import re

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    WINDOW_SIZE,
    minmax_normalize,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)

# Configure multi-core usage
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

def main(stock, window_size=WINDOW_SIZE, batch_size=32, ep_count=50,
         strategy="t-dqn", model_name=None, pretrained=False,
         debug=False, model_type='dense'):
    import numpy as np
    """ Finetune the stock trading bot on a large interval (2019-01-01 — 2024-06-30).
    Logs each epoch to train_finetune.log, uses tqdm for progress.
    """
    import pandas as pd
    import os
    import logging
    from trading_bot.ops import get_state
    from tqdm import tqdm

    # Настраиваем логирование в файл
    logging.basicConfig(filename="train_finetune.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("Лог обучения будет писаться в train_finetune.log")

    # Загружаем данные 2019-01-01 — 2024-06-30
    # Clean CSV: keep only header and lines starting with date
    pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    raw_lines = []
    with open(stock, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Date,'):
                raw_lines.append(line)
            else:
                m = pattern.search(line)
                if m:
                    raw_lines.append(line[m.start():])
    df = pd.read_csv(io.StringIO('\n'.join(raw_lines)))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Split into train (2015–2023) and validation (2024-01-01–2024-06-30)
    train_df = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2024-01-01")]
    val_df = df[(df["Date"] >= "2024-01-01") & (df["Date"] <= "2024-06-30")]
    raw_train_prices = train_df["Adj Close"].values
    raw_train_volumes = train_df["Volume"].values
    # normalize price and volume and combine
    price_norm = list(minmax_normalize(raw_train_prices))
    vol_norm = list(minmax_normalize(raw_train_volumes))
    train_data = list(zip(price_norm, vol_norm))
    val_price = list(minmax_normalize(raw_train_prices[len(train_df):]))
    val_vol = list(minmax_normalize(raw_train_volumes[len(train_df):]))
    val_data = list(zip(val_price, val_vol))
    print(f"Train: {len(train_df)} days")
    print(f"train_data: min={np.min(train_data):.2f}, max={np.max(train_data):.2f}, mean={np.mean(train_data):.2f}")
    # Принудительно создать лог-файл
    with open("train_finetune.log", "a") as f: f.write("=== Training started ===\n")

    # Resolve pretrained model path: normalize to basename
    import glob
    if pretrained:
        # find latest .h5 in models/ if no specific file
        if not model_name or not model_name.endswith('.h5'):
            h5_files = glob.glob(os.path.join('models', '*.h5'))
            if not h5_files:
                logging.warning('No pretrained .h5 found in models/, training from scratch.')
                pretrained = False
                model_name = None
            else:
                h5_files.sort(key=os.path.getmtime, reverse=True)
                model_name = os.path.basename(h5_files[0])
        else:
            # keep only filename
            model_name = os.path.basename(model_name)

    # Initialize agent with dynamic window_size (state_size computed internally)
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    agent.model_type = model_type
    if model_type == 'lstm' and model_name and not model_name.endswith('_LSTM'):
        model_name += '_LSTM'
        agent.model_name = model_name

    best_val_profit = None
    best_val_epoch = None
    no_improve = 0
    strategies = strategy.split(",")
    for strat in strategies:
        agent.strategy = strat
    for epoch in tqdm(range(1, ep_count+1), desc="Finetune Epoch"):
        result = train_model(agent, epoch, train_data, ep_count=ep_count, batch_size=batch_size, window_size=window_size)
        # Оценим на трейне (для контроля)
        agent.epsilon = 0.0
        state = get_state(train_data, 0, window_size)
        profit = 0
        buy_count = 0
        sell_count = 0
        position = []
        for t in range(len(train_data)):
            action = agent.act(state, is_eval=True)
            next_state = get_state(train_data, t+1, window_size) if t+1 < len(train_data) else state
            if action == 1:
                buy_count += 1
                position.append(raw_train_prices[t])
            elif action == 2 and len(position) > 0:
                sell_count += 1
                buy_price = position.pop(0)
                profit += raw_train_prices[t] - buy_price
            state = next_state
        # liquidate remaining positions at last price
        for buy_price in position:
            profit += raw_train_prices[-1] - buy_price
        print(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={buy_count+sell_count}")
        logging.info(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={buy_count+sell_count}")
        with open("train_finetune.log", "a") as f:
            f.write(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]} trades={buy_count+sell_count}\n")
        # Evaluate on val set
        val_profit, _ = evaluate_model(agent, val_data, window_size, debug, min_v=np.min(raw_train_prices), max_v=np.max(raw_train_prices))
        logging.info(f"Epoch {epoch}/{ep_count}: val_profit={val_profit:.2f}")
        if best_val_profit is None or val_profit > best_val_profit:
            best_val_profit = val_profit
            best_val_epoch = epoch
            agent.model.save_weights(f"models/best_{strat}_{model_name}_{window_size}.h5")
            no_improve = 0
        else:
            no_improve += 1
        if earlystop_patience and no_improve >= earlystop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")
    logging.info(f"Best val profit={best_val_profit:.2f} at epoch {best_val_epoch}")


if __name__ == "__main__":
    args = docopt(__doc__)

    stock = args["<stock>"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    strategy = args["--strategy"]
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    model_type = args.get("--model-type") or 'dense'

    # LSTM: 500 эпох, earlystop=50
    if model_type == 'lstm':
        ep_count = 500
        earlystop_patience = 50
    else:
        earlystop_patience = None

    main(stock, window_size=window_size, batch_size=batch_size, ep_count=ep_count,
         strategy=strategy, model_name=model_name, pretrained=pretrained, debug=debug, model_type=model_type)
