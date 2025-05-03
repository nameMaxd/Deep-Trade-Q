# Deep-Trade-AI---Q-Learning-Trading-Bot


# Overview

This project implements a Stock Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-learning. Implementation is kept simple and as close as possible to the algorithm discussed in the paper, for learning purposes.

## Introduction

Generally, Reinforcement Learning is a family of machine learning techniques that allow us to create intelligent agents that learn from the environment by interacting with it, as they learn an optimal policy by trial and error. This is especially useful in many real world tasks where supervised learning might not be the best approach due to various reasons like nature of task itself, lack of appropriate labelled data, etc.

The important idea here is that this technique can be applied to any real world task that can be described loosely as a Markovian process.

## Approach

This work uses a Model-free Reinforcement Learning technique called Deep Q-Learning (neural variant of Q-Learning).
At any given time (episode), an agent abserves it's current state (n-day window stock price representation), selects and performs an action (buy/sell/hold), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts it's parameters based on the gradient of the loss computed.



## Results

Trained on `GOOG` 2010-17 stock data, tested on 2019 with a profit of $1141.45 with a model accuracy of 89.98% :

![Google Stock Trading episode](./extra/visualization.png)

You can obtain similar visualizations of your model evaluations using the [notebook](./visualize.ipynb) provided.



## Data

You can download Historical Financial data from [Yahoo! Finance](https://ca.finance.yahoo.com/) for training, or even use some sample datasets already present under `data/`.

## Getting Started

In order to use this project, you'll need to install the required python packages:

```bash
pip3 install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

```bash
python3 train.py data/GOOG.csv data/GOOG_2018.csv --strategy t-dqn
```

Once you're done training, run the evaluation script and let the agent make trading decisions:

```bash
python3 eval.py data/GOOG_2019.csv --model-name model_GOOG_50 --debug
```

### TD3 Support
You can now train with TD3 via the same CLI:
```bash
python3 train.py data/GOOG_2010-2024-06.csv --strategy td3 \
  --window-size 50 --td3-timesteps 200000 --td3-noise-sigma 0.2 \
  --td3-save-name my_td3_GOOG
```
The model will be saved as `<td3-save-name>_<stock>.zip`.  
Note: TD3 requires `stable_baselines3` and `gymnasium`. Install them with:
```bash
pip3 install stable_baselines3 gymnasium
```

Now you are all set up!

## Acknowledgements

- [@keon](https://github.com/keon) for [deep-q-learning](https://github.com/keon/deep-q-learning)
- [@edwardhdlu](https://github.com/edwardhdlu) for [q-trader](https://github.com/edwardhdlu/q-trader)

## References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)


---
# [Автоматически извлечённая статья](2208.07165v1.pdf)
# Deep Reinforcement Learning Approach for Trading Automation in The Stock Market

**Авторы:** Taylan Kabbani, Ekrem Duman, Ozyegin University, Istanbul, Turkey

**Контакты:** 1taylan.kabbani1@ozu.edu.tr, 2ekrem.duman@ozyegin.edu.tr

---

## Краткое содержание
- Использование DRL (Deep Reinforcement Learning) для автоматизации трейдинга на фондовом рынке.
- Формулировка задачи как POMDP с учетом ограничений рынка (ликвидность, комиссии).
- Решение с помощью TD3 (Twin Delayed Deep Deterministic Policy Gradient).
- Достижение коэффициента Шарпа 2.68 на тестовой выборке.

---

## Ключевые слова
Автономный агент, глубокое обучение с подкреплением, MDP, анализ новостей, технические индикаторы, TD3.

---

## Основные разделы статьи

### 1. Введение
Задача: минимизировать риски и максимизировать прибыль на рынке. DRL позволяет объединить предсказание цен и аллокацию капитала в единую систему.

### 2. Теория RL и MDP
- MDP и POMDP: формализация задачи трейдинга.
- Bellman equations:

$$
V^*(s) = \max_a \sum_{s', r} P(s', r | s, a)[r + \gamma V^*(s')]
$$

- RL-алгоритмы: Critic-only, Actor-only, Actor-Critic.

### 3. Формализация задачи трейдинга
- Состояние: вектор (1 + 13*N), где N — число активов.
- Действие: портфельная аллокация (buy/sell/hold), непрерывное пространство.
- Вознаграждение: разница портфельной стоимости между шагами.
- Ограничения: отсутствие short, комиссия, нет отрицательного баланса, нет влияния на рынок.

### 4. TD3-алгоритм
- Улучшение DDPG:
    1. Clipped Double Critic Networks
    2. Delayed Updates
    3. Target Policy Smoothing

#### Pseudocode TD3
```latex
1. Инициализация Q(s,a|w_1), Q(s,a|w_2), π(s|θ)
2. Для каждого шага t:
    - a ~ π(s|θ) + noise
    - y = r + γ * min(Q1', Q2')
    - Обновить критики и актор
    - Обновить таргет-неты
```

### 5. Данные и препроцессинг
- Исторические цены Yahoo Finance, финансовые новости (Benzinga, Reddit, Kaggle).
- Технические индикаторы: RSI, SMA, EMA, MACD, OBV, и др.
- Sentiment анализ: FinBERT.

### 6. Эксперименты
- Базовый вариант: только цены — Sharpe 1.43
- + Тех. индикаторы — Sharpe 2.75
- + Sentiment — Sharpe 3.14
- Тест на 10 активах — Sharpe 2.68

#### Таблица результатов
| Окружение           | Доходность      | Sharpe | Комиссия |
|---------------------|----------------|--------|----------|
| Baseline            | 33960$±4473    | 1.43   | 355$±83  |
| WithTechIndicators  | 89782$±18980   | 2.75   | 1109$±248|
| WithSentiments      | 115591$±17721  | 3.14   | 1447$±268|
| Benchmark (Kaur)    | —              | 0.85   | —        |

---

## Основные формулы
- Bellman Optimality:

$$
V^*(s) = \max_a \sum_{s', r} P(s', r | s, a)[r + \gamma V^*(s')]
$$

- Actor-Critic update:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi(a_t|s_t, \theta_t) (R_{t+1} + \gamma \hat{V}(s_{t+1}, w) - \hat{V}(s_t, w))
$$

- Reward:

$$
r(s, a, s') = V_t - V_{t-1}
$$

- Portfolio value:

$$
V_t = b_t + h_t \cdot C_t
$$

---

## Выводы
- DRL и TD3 позволяют строить прибыльные стратегии на рынке акций.
- Добавление тех. индикаторов и анализа новостей улучшает результат.
- Модель устойчива и обобщается на новые данные.

---

## Ссылки и литература
(см. оригинальный список в статье)

---

_Файл собран автоматически из PDF. Для подробностей смотри оригинал/2208.07165v1.pdf._
