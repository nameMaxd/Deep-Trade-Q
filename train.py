"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    WINDOW_SIZE,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def main(train_stock, val_stock, window_size=WINDOW_SIZE, batch_size=32, ep_count=50,
         strategy="t-dqn", model_name=None, pretrained=False,
         debug=False):
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
    df = pd.read_csv('data/GOOG.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    train_df = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2024-01-01")]
    val_df = df[(df["Date"] >= "2024-01-01") & (df["Date"] < "2025-01-01")]
    test_df = df[(df["Date"] >= "2025-01-01")]
    train_data = get_stock_data(train_stock, norm_type="minmax")
    val_data = get_stock_data(val_stock, norm_type="minmax")
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days, Test: {len(test_df)} days")
    print(f"train_data: min={np.min(train_data):.2f}, max={np.max(train_data):.2f}, mean={np.mean(train_data):.2f}")
    # Принудительно создать лог-файл
    with open("train_finetune.log", "a") as f: f.write("=== Training started ===\n")

    # Загружаем агент с весами
    if model_name is None:
        model_name = "model_t-dqn_GOOG_10_FEATS_2020-2024-06"
    state_size = window_size - 1 + 3
    agent = Agent(state_size, strategy=strategy, pretrained=pretrained, model_name=model_name)

    best_profit = None
    best_epoch = None
    for epoch in tqdm(range(1, ep_count+1), desc="Finetune Epoch"):
        result = train_model(agent, epoch, train_data, ep_count=ep_count, batch_size=batch_size, window_size=window_size)
        # Оценим на трейне (для контроля)
        agent.epsilon = 0.0
        state = get_state(train_data, 0, window_size)
        profit = 0
        position = []
        for t in range(len(train_data)):
            action = agent.act(state, is_eval=True)
            next_state = get_state(train_data, t+1, window_size) if t+1 < len(train_data) else state
            if action == 1:
                position.append(train_data[t])
            elif action == 2 and len(position) > 0:
                buy_price = position.pop(0)
                profit += train_data[t] - buy_price
            state = next_state
        print(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]}")
        logging.info(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]}")
        with open("train_finetune.log", "a") as f:
            f.write(f"Epoch {epoch}/{ep_count}: train_profit={profit:.2f} train_loss={result[3]}\n")
        if best_profit is None or profit > best_profit:
            best_profit = profit
            best_epoch = epoch
            # Сохраняем лучшие веса
            agent.model.save_weights("models/model_t-dqn_GOOG_10_finetuned.h5")
    print(f"Лучший train profit={best_profit:.2f} на эпохе {best_epoch}. Веса сохранены в models/model_t-dqn_GOOG_10_finetuned.h5")
    with open("train_finetune.log", "a") as f:
        f.write(f"Лучший train profit={best_profit:.2f} на эпохе {best_epoch}.\n")


if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
