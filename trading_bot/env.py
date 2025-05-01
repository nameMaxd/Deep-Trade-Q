import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size=47, commission=0.001, max_inventory=8, carry_cost=0.0001, min_trade_value=10.0, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True):
        super().__init__()
        self.prices = prices
        self.volumes = volumes
        # position limits and holding costs
        self.max_inventory = max_inventory
        self.carry_cost = carry_cost
        # global normalization bounds override
        self.global_min_v = min_v
        self.global_max_v = max_v
        self.window_size = window_size
        # features: window_size-1 sigmoids + SMA, EMA, RSI, vol_ratio + momentum, volatility
        self.state_size = window_size - 1 + 6  # без +2, inventory добавляется отдельно!
        # transaction commission fraction
        self.commission = commission
        # minimum trade value threshold (in $) to open a position
        self.min_trade_value = 1000.0  # Увеличиваем размер сделки до $1000 для более заметного профита
        # risk aversion parameters
        self.risk_lambda = risk_lambda
        self.drawdown_lambda = drawdown_lambda
        self.dual_phase = dual_phase
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([2.0]), shape=(1,), dtype=np.float32
        )
        # observation includes inventory features, unbounded
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size + 2,), dtype=np.float32)
        print(f"[env] window_size={window_size} state_size={self.state_size} obs_space={self.observation_space.shape}")
        
        # Инициализация счетчиков действий для отслеживания повторяющихся действий
        self.action_counter = {0: 0, 1: 0, 2: 0}
        self.last_action = 0

    def reset(self, *args, random_start=False, **kwargs):
        # Accept arbitrary seed/options args and handle random_start correctly
        super().reset(*args, **kwargs)
        # Set starting step
        if random_start:
            self.current_step = np.random.randint(0, len(self.prices) - self.window_size - 1)
        else:
            self.current_step = 0
        # Debug log for first few resets
        if not hasattr(self, '_reset_log_count'):
            self._reset_log_count = 0
        # Не логируем ничего, чтобы не мешать tqdm
        self.inventory = []
        self.total_profit = 0.0
        # initialize risk tracking
        self.rewards = []
        self.equity = [0.0]
        self.max_equity = 0.0
        self.last_action_step = None  # Для отслеживания последнего действия
        # Счетчик сделок для статистики
        self.trade_count = 0
        # choose episode phase
        if self.dual_phase:
            self.phase = 'exploration' if np.random.rand() < 0.5 else 'exploitation'
        else:
            self.phase = 'exploitation'
        # Умеренный штраф за удержание, чтобы не перегружать модель
        self.hold_penalty = 0.05  # Умеренный штраф за удержание
        # Сбрасываем счетчики действий
        self.action_counter = {0: 0, 1: 0, 2: 0}
        self.last_action = 0
        # global normalization bounds if provided, else compute from data
        if self.global_min_v is not None:
            self.min_v = self.global_min_v
        else:
            self.min_v = np.min(self.prices)
        if self.global_max_v is not None:
            self.max_v = self.global_max_v
        else:
            self.max_v = np.max(self.prices)
        state_arr = get_state(
            list(zip(self.prices, self.volumes)),
            self.current_step, self.window_size,
            min_v=self.min_v, max_v=self.max_v
        )
        # extend state with inventory features (initially zero)
        inv_count_norm = 0.0
        avg_entry_ratio = 0.0
        state_ext = np.concatenate([state_arr, [inv_count_norm, avg_entry_ratio]]).astype(np.float32)
        return state_ext, {}

    def _calculate_reward(self, action):
        """Рассчитывает вознаграждение на основе действия и изменения цены"""
        step_reward = 0.0
        
        # Записываем шаг последнего действия для отслеживания
        if action != 0:  # Если действие не "держать"
            self.last_action_step = self.current_step
        
        # Вознаграждение за изменение цены
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        prev_price = self.prices[max(0, min(self.current_step - 1, len(self.prices) - 1))]
        price_change = (current_price - prev_price) / prev_price
        
        # ЭКСТРЕМАЛЬНО агрессивное вознаграждение за действия
        if action == 1:  # Покупка
            # Огромное вознаграждение за любую покупку для стимулирования торговли
            step_reward = 10.0
            # Дополнительный бонус за покупку при росте цены
            if price_change > 0:
                step_reward += 5.0 + price_change * 500
        elif action == 2:  # Продажа
            # Огромное вознаграждение за любую продажу для стимулирования торговли
            step_reward = 10.0
            # Дополнительный бонус за продажу при падении цены
            if price_change < 0:
                step_reward += 5.0 - price_change * 500
        else:  # Держать
            # Огромный штраф за бездействие, чтобы заставить агента торговать
            step_reward = -10.0
            
            # Увеличивающийся штраф, если давно не было действий
            if self.last_action_step is not None:
                inactivity_penalty = (self.current_step - self.last_action_step) * 0.5
                step_reward -= min(inactivity_penalty, 20.0)  # Ограничиваем максимальный штраф
        
        # Дополнительное вознаграждение за смену действий (чтобы агент не зацикливался на одном действии)
        if self.last_action != action and action != 0:
            step_reward += 5.0  # Большой бонус за смену стратегии
        
        # Штраф за повторение одного и того же действия много раз подряд
        if action == 0 and self.action_counter[action] > 2:  # Быстро наказываем за удержание
            step_reward -= 2.0 * (self.action_counter[action] - 2)
        elif action != 0 and self.action_counter[action] > 10:  # Позволяем больше повторений для торговых действий
            step_reward -= 1.0 * (self.action_counter[action] - 10)
        
        # Обновляем счетчик действий
        for a in range(3):  # 0, 1, 2 - все возможные действия
            if a == action:
                self.action_counter[a] = self.action_counter.get(a, 0) + 1
            else:
                self.action_counter[a] = 0
        
        # Сохраняем последнее действие для следующего шага
        self.last_action = action
        
        return step_reward

    def step(self, action):
        # map continuous to discrete action: 0=HOLD,1=BUY,2=SELL
        if isinstance(action, (list, np.ndarray)):
            a = float(action[0])
        else:
            a = float(action)
        
        # Принудительно заставляем агента торговать
        # С вероятностью 90% заменяем HOLD на торговое действие
        if a < 0.3 and np.random.random() < 0.9:
            # Заменяем HOLD на BUY или SELL случайным образом
            a = np.random.choice([1.0, 2.0])
            
        action = int(np.clip(np.round(a), 0, 2))

        # Проверяем, что индекс не выходит за границы массива
        safe_step = min(self.current_step, len(self.prices) - 1)
        price = float(self.prices[safe_step])

        # Получаем вознаграждение на основе действия и изменения цены
        reward = self._calculate_reward(action)

        # Обрабатываем действия агента
        if action == 0:  # HOLD
            # Штраф за удержание уже учтен в _calculate_reward
            pass
            
        elif action == 1:  # BUY
            # BUY: invest fixed amount self.min_trade_value (fractional share)
            if len(self.inventory) < self.max_inventory:
                qty = self.min_trade_value / price
                self.inventory.append((price, qty))
                # Записываем факт совершения сделки
                self.last_action_step = self.current_step
                # Увеличиваем счетчик сделок
                self.trade_count += 1
                reward -= self.commission * price * qty
            else:
                # Если инвентарь полон, применяем штраф за бездействие
                reward -= self.hold_penalty
                
        elif action == 2:  # SELL
            # SELL
            if self.inventory:
                bought_price, qty = self.inventory.pop(0)
                profit = (price - bought_price) * qty
                cost = self.commission * (price * qty + bought_price * qty)
                net = profit - cost
                reward += net
                self.total_profit += net
                # Записываем факт совершения сделки
                self.last_action_step = self.current_step
                # Увеличиваем счетчик сделок
                self.trade_count += 1
            else:
                # Если нечего продавать, применяем штраф за бездействие
                reward -= self.hold_penalty
        
        # carry cost per item in inventory
        reward -= self.carry_cost * len(self.inventory)
        
        # mark-to-market reward for inventory due to price change
        if self.current_step < len(self.prices) - 1 and self.inventory:
            next_step = min(self.current_step + 1, len(self.prices) - 1)
            next_price = float(self.prices[next_step])
            mtm = sum((next_price - price) * qty for price, qty in self.inventory)
            reward += mtm
            
        # update risk metrics
        self.rewards.append(reward)
        self.equity.append(self.equity[-1] + reward)
        self.max_equity = max(self.max_equity, self.equity[-1])
        vol = float(np.std(self.rewards)) if len(self.rewards) > 1 else 0.0
        drawdown = float(self.max_equity - self.equity[-1])
        
        if self.dual_phase and self.phase == 'exploration':
            # exploration phase: focus on risk minimization
            reward = - self.risk_lambda * vol - self.drawdown_lambda * drawdown
        else:
            # exploitation phase or no dual phase: profit minus risk penalties
            reward = reward - self.risk_lambda * vol - self.drawdown_lambda * drawdown
            
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Liquidate remaining inventory at end of episode (market close)
        if self.current_step >= len(self.prices) - 2 and self.inventory:
            # Проверяем, что индекс не выходит за границы массива
            safe_step = min(self.current_step, len(self.prices) - 1)
            final_price = float(self.prices[safe_step])
            for bought_price, qty in self.inventory:
                profit = (final_price - bought_price) * qty
                cost = self.commission * (final_price * qty + bought_price * qty)
                net = profit - cost
                reward += net
                self.total_profit += net
            self.inventory = []
            
        # get state array - используем безопасный индекс
        safe_step = min(self.current_step, len(self.prices) - 1)
        base_state = get_state(
            list(zip(self.prices, self.volumes)),
            safe_step, self.window_size,
            min_v=self.min_v, max_v=self.max_v
        )
        
        # extend state with inventory info
        inv_count_norm = len(self.inventory) / self.max_inventory
        if self.inventory:
            total_qty = sum(qty for _, qty in self.inventory)
            avg_price = sum(bp * qty for bp, qty in self.inventory) / total_qty
            # Используем безопасный индекс для текущего шага
            safe_step = min(self.current_step, len(self.prices) - 1)
            current_price = float(self.prices[safe_step])
            avg_entry_ratio = (current_price - avg_price) / avg_price
            avg_entry_ratio = np.clip(avg_entry_ratio, -1.0, 1.0)
        else:
            avg_entry_ratio = 0.0
            
        state_ext = np.concatenate([base_state, [inv_count_norm, avg_entry_ratio]]).astype(np.float32)
        obs = state_ext
        
        # info теперь содержит реальное действие
        info = {'real_action': action}
        return obs, reward, done, False, info
