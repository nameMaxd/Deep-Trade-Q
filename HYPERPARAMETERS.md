# Hyperparameters Overview

В этом файле описаны все гиперпараметры проекта Deep-Trade-Q, их роль, влияние и типичные диапазоны значений.

---

## 1. Agent configuration (`trading_bot/agent.py`)

- `window_size` (int)
  - Кол-во прошлых временных шагов во входном состоянии.
  - Default: 50. ↑ контекст, ↑ вычисления; ↓ — наоборот.

- `state_size` (int)
  - Вычисляется как `window_size - 1 + 4` (+ признаки SMA, EMA, RSI, vol_ratio).

- `action_size` (int)
  - Число действий: 3 (0=Hold, 1=Buy, 2=Sell).

- `strategy` (str)
  - Тип DQN: "dqn", "t-dqn" (target network), "double-dqn".
  - Определяет формулу обновления Q и частоту синхронизации target_model.

- `reset_every` / `target_update` (int)
  - После N итераций синхронизировать target_model.
  - Default: 5000.

- `n_iter` (int)
  - Счётчик шагов для `reset_every`.

- `memory.maxlen` (int)
  - Размер replay buffer.
  - Default: 10000. Меньше — быстрее выборка, но меньше разнообразия.

- `gamma` (float)
  - Discount-фактор будущей награды.
  - Default: 0.95. ↑ — внимание на долгосрочные reward; ↓ — краткосрочные.

- `epsilon` (float)
  - Начальное ε для ε-greedy.
  - Default: 1.0.

- `epsilon_decay` (float)
  - Умножение ε после каждой эпохи.
  - Default: 0.995. Меньше — медленнее спад, больше exploration.

- `epsilon_min` (float)
  - Нижний порог ε.
  - Default: 0.01.

- `tau` (float)
  - Температура для Boltzmann-сэмплинга при обучении.
  - Default: 1.0. ↑τ — более случайный выбор; ↓τ — ближе к greedy.

- `learning_rate` (float)
  - Шаг обучения для Adam optimizer.
  - Default: 0.001. ↓ — стабильнее, дольше; ↑ — быстрее, может быть неустойчивым.

- `optimizer.clipnorm` (float)
  - Clip-norm градиентов.
  - Default: 1.0. ↓ — жестче клиппинг.

- `loss` (callable)
  - Функция потерь: `huber_loss`. Менее чувствительна к выбросам.

- `custom_objects` (dict)
  - Нужны при загрузке модели (`{'huber_loss': huber_loss}`).

- `pretrained` (bool)
  - Загрузка ранее обученной модели.

- `model_name` (str)
  - Путь/имя файла модели для load/save.

---

## 2. Neural Network Architecture

### 2.1 Dense (default)
- `input_dim = state_size`.
- Слои:
  1. Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
  2. BatchNormalization()
  3. Dropout(0.2)
  4. Dense(256, activation='relu', kernel_regularizer=l2(1e-4))
  5. BatchNormalization()
  6. Dropout(0.3)
  7. Dense(256, activation='relu', kernel_regularizer=l2(1e-4))
  8. BatchNormalization()
  9. Dropout(0.3)
  10. Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
  11. BatchNormalization()
  12. Dropout(0.2)
  13. Dense(action_size, activation='linear')
- **Dropout**: 0.2–0.3. ↓ — меньше регуляризация.
- **L2**: 1e-4. ↑ — сильнее штраф за сложные веса.

### 2.2 LSTM (`--model-type=lstm`)
- `Reshape((state_size,1))` → вход в LSTM.
- LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)), BatchNorm(), Dropout(0.2)
- LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-4)), BatchNorm(), Dropout(0.3)
- LSTM(256, return_sequences=False, kernel_regularizer=l2(1e-4)), BatchNorm(), Dropout(0.3)
- Dense(128, activation='relu', kernel_regularizer=l2(1e-4)), BatchNorm(), Dropout(0.2)
- Dense(action_size, activation='linear')

---

## 3. Training parameters (`train.py`, `trading_bot/methods.py`)

- `batch_size` (int)
  - Default: 32. ↑ — более стабильная оценка градиента; ↓ — быстрее шаг.

- `ep_count` / `epochs` (int)
  - Число эпох. Default: 500. Для быстрых тестов 50–100.

- `window_size` (int)
  - Должен совпадать с Agent.window_size.

- `replay_freq` (int)
  - Частота вызова train_experience_replay (по шагам).
  - Default: 100.

- `earlystop_patience` (int)
  - Патience для EarlyStopping на валидации.
  - Default: 50 (для LSTM).

- `min_v`, `max_v` (float)
  - Нормализация цен в методах `get_state`.

---

## 4. Risk Management (`trading_bot/env.py`)

- `stop_loss_pct` (float)  
  - Процент максимального убытка для автоматического закрытия позиции.  
  - Default: 0.02 (2%). ↑ — больше риска; ↓ — меньше просадки.

---

## 5. Reward shaping (`trading_bot/methods.py`)

- `profit_weight` (`delta * 10.0`)
  - Усиленный стимул за прибыль.

- `loss_penalty` (`delta * -0.5`)

- `sell_noinv_penalty` (`-0.02`)

- `hold_penalty_per_step` (`-0.005 * len(inventory)`) 

- `hold_noinv_penalty` (`-0.01`)

---

## 6. Data features & preprocessing

- Признаки: цена, объём, SMA, EMA, RSI, volatility ratio.
- Сплит: train/val (по умолчанию ~80/20).

---

## 7. Дополнительные гипотезы

- Mixed-precision (`mixed_float16`).
- Prioritized Replay.
- Policy-gradient методы (PPO, A2C).
- Hyperparameter tuning (Optuna, Hyperopt).
