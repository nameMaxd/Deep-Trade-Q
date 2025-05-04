"""
Централизованный конфиг для всех параметров проекта
"""

# ===== Trading Parameters =====
WINDOW_SIZE = 47           # Размер окна для анализа
COMMISSION = 0.001         # Комиссия за сделку (0.1%)
MIN_TRADE_VALUE = 1000.0   # Минимальный размер сделки ($)
MAX_INVENTORY = 8          # Макс. количество позиций
CARRY_COST = 0.0001        # Стоимость удержания позиции
STOP_LOSS_PCT = 0.05       # Автоматическое закрытие при убытке >5%

# ===== Risk Management =====
RISK_LAMBDA = 0.1          # Коэффициент aversion к риску
DRAWDOWN_LAMBDA = 0.1      # Штраф за просадку

# ===== Volume Profile =====
VOLUME_BINS = 10           # Количество уровней объема
VOLUME_LOOKBACK = 50       # Глубина анализа объема (баров)

# ===== RL Training =====
# Основные параметры
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_DECAY = 0.995
BATCH_SIZE = 32            # Размер батча
EPISODE_COUNT = 50         # Количество эпизодов
TARGET_UPDATE = 100        # Частота обновления target сети

# ===== TD3 Hyperparameters =====
TD3_NET_ARCH = [256, 256, 128]      # Архитектура сети
TD3_TRAIN_FREQ = (5, 'step')        # Частота обучения (каждые 5 шагов)
TD3_GRADIENT_STEPS = 5              # Шагов градиента за обновление
TD3_LEARNING_STARTS = 1000          # Шумов до начала обучения  
TD3_POLICY_DELAY = 2                # Задержка обновления политики
TD3_SEED = 42                       # Фиксированный seed
TD3_TAU = 0.005                     # Коэффициент обновления target network
TD3_VERBOSE = 1                     # Уровень логгирования

# TD3 специфичные
TD3_TIMESTEPS = 100000     # Общее количество шагов
TD3_POLICY_NOISE = 0.7
TD3_NOISE_CLIP = 0.5
TD3_NOISE_SIGMA = 0.1  # Уровень шума для exploration
TD3_SAVE_NAME = 'td3_model' # Имя для сохранения модели

# ===== Environment =====
INITIAL_BALANCE = 10000.0  # Стартовый депозит
DUAL_PHASE = True          # Двухфазное обучение
# MODEL_TYPE = 'dense'       # Тип модели: 'dense' или 'lstm'
