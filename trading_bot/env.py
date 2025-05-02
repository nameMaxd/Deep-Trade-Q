import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size, commission=0.001, max_inventory=8, carry_cost=0.0001, min_trade_value=10.0, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True):
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
        self.state_size = window_size - 1 + 6 + 2  # +2 inventory features
        # transaction commission fraction
        self.commission = commission
        # minimum trade value threshold (in $) to open a position
        self.min_trade_value = min_trade_value
        # risk aversion parameters
        self.risk_lambda = risk_lambda
        self.drawdown_lambda = drawdown_lambda
        self.dual_phase = dual_phase
        # Для TD3 используем непрерывное пространство действий [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32
        )
        # observation includes inventory features, unbounded
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)

    def reset(self, *args, seed=None, options=None, random_start=False, **kwargs):
        # Важно: возвращаем кортеж (observation, info)
        # Это критично для совместимости с SB3 на Windows
        super().reset(seed=seed)
        if random_start:
            self.current_step = np.random.randint(0, len(self.prices) - self.window_size - 1)
        else:
            self.current_step = 0
        if not hasattr(self, '_reset_log_count'):
            self._reset_log_count = 0
        self.inventory = []
        self.total_profit = 0.0
        # initialize risk tracking
        self.rewards = []
        self.equity = [0.0]
        self.max_equity = 0.0
        # choose episode phase
        if self.dual_phase:
            self.phase = 'exploration' if np.random.rand() < 0.5 else 'exploitation'
        else:
            self.phase = 'exploitation'
        # penalty for holding to force actions
        self.hold_penalty = 0.01  # reduced hold penalty to discourage unnecessary holds
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
        )[0]
        inv_count_norm = 0.0
        avg_entry_ratio = 0.0
        state_ext = np.concatenate([state_arr, [inv_count_norm, avg_entry_ratio]]).astype(np.float32)
        # Возвращаем кортеж (observation, info)
        return state_ext, {}

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            a = float(action[0])
        else:
            a = float(action)
        
        # Преобразуем непрерывное действие в дискретное
        # 0.0-0.33 -> HOLD (0)
        # 0.34-0.66 -> BUY (1)
        # 0.67-1.0 -> SELL (2)
        if a < 0.33:
            action = 0  # HOLD
        elif a < 0.66:
            action = 1  # BUY
        else:
            action = 2  # SELL
        
        price = float(self.prices[self.current_step])
        reward = 0.0
        
        # Для отладки
        action_taken = action
        
        # Штраф за бездействие - заставляем агента совершать сделки
        if action == 0:  # HOLD
            # Увеличиваем штраф за удержание с каждым шагом
            hold_steps = getattr(self, 'consecutive_holds', 0) + 1
            setattr(self, 'consecutive_holds', hold_steps)
            # Экспоненциальный штраф за длительное удержание
            reward -= self.hold_penalty * (1.0 + 0.01 * hold_steps)
        else:
            # Сбрасываем счетчик удержаний при активном действии
            setattr(self, 'consecutive_holds', 0)
        
        if action == 1:  # BUY
            if len(self.inventory) < self.max_inventory:
                qty = self.min_trade_value / price
                self.inventory.append((price, qty))
                reward += 0.5  # Значительный бонус за открытие позиции
                reward -= self.commission * price * qty
            else:
                # Если инвентарь полон, считаем как HOLD
                action = 0
                reward -= self.hold_penalty
        elif action == 2:  # SELL
            if self.inventory:
                bought_price, qty = self.inventory.pop(0)
                profit = (price - bought_price) * qty
                cost = self.commission * (price * qty + bought_price * qty)
                net = profit - cost
                
                # Усиливаем сигнал награды для прибыльных сделок
                if net > 0:
                    reward += net * 3.0  # Утраиваем положительную прибыль
                else:
                    reward += net * 0.5  # Уменьшаем отрицательную прибыль
                    
                self.total_profit += net
            else:
                # Если нечего продавать, считаем как HOLD
                action = 0
                reward -= self.hold_penalty * 3  # Сильный штраф за попытку продать без позиций
        
        # Небольшой штраф за хранение позиций
        reward -= self.carry_cost * len(self.inventory)
        
        # Учитываем потенциальную прибыль/убыток от открытых позиций
        if self.current_step < len(self.prices) - 1 and self.inventory:
            next_price = float(self.prices[self.current_step + 1])
            mtm = sum((next_price - price) * qty for price, qty in self.inventory)
            
            # Поощряем держать выигрышные позиции и закрывать проигрышные
            if mtm > 0:
                reward += mtm * 0.3  # Меньший вес для потенциальной прибыли
            else:
                reward += mtm * 0.1  # Еще меньший вес для потенциальных убытков
        
        # update risk metrics
        self.rewards.append(reward)
        self.equity.append(self.equity[-1] + reward)
        self.max_equity = max(self.max_equity, self.equity[-1])
        
        # Расчет Sharpe Ratio на основе последних доходностей
        # Используем формулу из статьи: r_t = Profit - λ·Risk
        # где Risk - это волатильность доходности (стандартное отклонение)
        returns = self.rewards[-20:] if len(self.rewards) > 20 else self.rewards
        
        if len(returns) > 1:
            # Волатильность доходности (риск)
            vol = float(np.std(returns))
            # Средняя доходность
            mean_return = float(np.mean(returns))
            # Sharpe Ratio (без безрисковой ставки)
            sharpe = mean_return / (vol + 1e-8)
            
            # Финальная награда: прибыль минус штраф за риск
            # r_t = Profit - λ·Risk
            reward = reward - self.risk_lambda * vol
        else:
            # Если недостаточно данных для расчета Sharpe
            reward = reward
        
        # ======= CRITICAL DEBUG BLOCK =========
        # Убираем форсированный done, возвращаем обычную логику
        done = self.current_step >= len(self.prices) - 2
        self.current_step += 1
        # Liquidate remaining inventory at end of episode (market close)
        # Исправлено: не выходим за пределы массива
        if self.current_step >= len(self.prices) - 2 and self.inventory:
            safe_step = min(self.current_step, len(self.prices) - 1)
            final_price = float(self.prices[safe_step])
            for bought_price, qty in self.inventory:
                profit = (final_price - bought_price) * qty
                cost = self.commission * (final_price * qty + bought_price * qty)
                net = profit - cost
                reward += net
                self.total_profit += net
            self.inventory = []
        # get state array
        base_state = get_state(
            list(zip(self.prices, self.volumes)),
            self.current_step, self.window_size,
            min_v=self.min_v, max_v=self.max_v
        )[0]
        # extend state with inventory info
        inv_count_norm = len(self.inventory) / self.max_inventory
        if self.inventory:
            total_qty = sum(qty for _, qty in self.inventory)
            avg_price = sum(bp * qty for bp, qty in self.inventory) / total_qty
            avg_entry_ratio = (float(self.prices[self.current_step]) - avg_price) / avg_price
            avg_entry_ratio = np.clip(avg_entry_ratio, -1.0, 1.0)
        else:
            avg_entry_ratio = 0.0
        state_ext = np.concatenate([base_state, [inv_count_norm, avg_entry_ratio]]).astype(np.float32)
        obs = state_ext
        info = {'real_action': action}
        assert isinstance(obs, np.ndarray) and obs.dtype == np.float32, f"obs type/shape: {type(obs)}, {obs.dtype}, {obs.shape}"
        assert isinstance(reward, float), f"reward type: {type(reward)}"
        assert isinstance(done, bool), f"done type: {type(done)}"
        return obs, reward, done, False, info
