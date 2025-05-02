import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size, commission=0.001, max_inventory=8, carry_cost=0.0001, min_trade_value=10.0, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True, max_episode_steps=None, hold_penalty=0.01, profit_bonus=0.2, loss_penalty=0.1):
        super().__init__()
        
        # Импортируем конфигурацию, если она не была импортирована
        try:
            from config import ENV_CONFIG
        except ImportError:
            # Значения по умолчанию, если конфигурация недоступна
            ENV_CONFIG = {
                "commission": 0.001,
                "max_inventory": 8,
                "carry_cost": 0.0001,
                "min_trade_value": 10.0,
                "risk_lambda": 0.1,
                "drawdown_lambda": 0.1,
                "dual_phase": True
            }
        
        self.prices = prices
        self.volumes = volumes
        
        # Используем параметры из аргументов или из конфигурации
        self.commission = commission if commission is not None else ENV_CONFIG.get("commission", 0.001)
        self.max_inventory = max_inventory if max_inventory is not None else ENV_CONFIG.get("max_inventory", 8)
        self.carry_cost = carry_cost if carry_cost is not None else ENV_CONFIG.get("carry_cost", 0.0001)
        self.min_trade_value = min_trade_value if min_trade_value is not None else ENV_CONFIG.get("min_trade_value", 10.0)
        self.risk_lambda = risk_lambda if risk_lambda is not None else ENV_CONFIG.get("risk_lambda", 0.1)
        self.drawdown_lambda = drawdown_lambda if drawdown_lambda is not None else ENV_CONFIG.get("drawdown_lambda", 0.1)
        self.dual_phase = dual_phase if dual_phase is not None else ENV_CONFIG.get("dual_phase", True)
        
        # global normalization bounds override
        self.global_min_v = min_v
        self.global_max_v = max_v
        self.window_size = window_size
        
        # features: window_size-1 sigmoids + SMA, EMA, RSI, vol_ratio + momentum, volatility
        self.state_size = window_size - 1 + 6 + 2  # +2 inventory features
        
        # Для TD3 используем непрерывное пространство действий [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32
        )
        
        # observation includes inventory features, unbounded
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        
        # Максимальное количество шагов в эпизоде = длина данных - размер окна
        # Это гарантирует, что один эпизод = один проход по всем данным
        self.max_steps = len(self.prices) - self.window_size
        
        # Новые параметры
        self.max_episode_steps = max_episode_steps or len(prices)
        self.hold_penalty = hold_penalty
        self.profit_bonus = profit_bonus
        self.loss_penalty = loss_penalty

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
        
        # Применяем действие
        if action == 1:  # BUY
            # Проверяем, не превышен ли лимит позиций
            if len(self.inventory) < self.max_inventory:
                # Проверяем, достаточно ли ценность сделки
                if price >= self.min_trade_value:
                    # Покупаем 1 акцию по текущей цене
                    self.inventory.append((price, 1.0))
                    # Комиссия как штраф
                    commission_cost = self.commission * price
                    reward -= commission_cost
                else:
                    # Слишком маленькая сделка, не покупаем
                    action = 0  # HOLD
            else:
                # Достигнут лимит позиций, не покупаем
                action = 0  # HOLD
                # Штраф за попытку превысить лимит
                reward -= 0.01
        
        elif action == 2:  # SELL
            # Проверяем, есть ли что продавать
            if self.inventory:
                # Продаем самую старую позицию (FIFO)
                bought_price, qty = self.inventory.pop(0)
                # Прибыль от продажи
                profit = (price - bought_price) * qty
                # Комиссия как процент от сделки
                commission_cost = self.commission * (price * qty + bought_price * qty)
                # Чистая прибыль
                net_profit = profit - commission_cost
                reward += net_profit
                
                # Добавляем бонус за прибыльную сделку или штраф за убыточную
                if profit > 0:
                    reward += profit * self.profit_bonus  # Бонус за прибыльную сделку
                else:
                    reward += profit * self.loss_penalty  # Меньший штраф за убыточную сделку
                
                self.total_profit += net_profit
            else:
                # Нечего продавать, не продаем
                action = 0  # HOLD
                # Штраф за попытку продать без позиций
                reward -= 0.01
        
        else:  # HOLD
            # Штраф за удержание (бездействие)
            reward -= self.hold_penalty
        
        # Штраф за удержание позиций (стоимость капитала)
        # Увеличиваем штраф в зависимости от количества шагов удержания
        if self.inventory:
            inventory_value = sum(bp * qty for bp, qty in self.inventory)
            # Увеличиваем штраф за удержание с каждым шагом
            holding_cost = self.carry_cost * inventory_value * (1.0 + self.current_step / 1000.0)
            reward -= holding_cost
        
        # Оценка потенциальной прибыли/убытка (mark-to-market)
        if self.current_step < len(self.prices) - 1 and self.inventory:
            next_price = float(self.prices[min(self.current_step + 1, len(self.prices) - 1)])
            mtm = sum((next_price - price) * qty for price, qty in self.inventory)
            
            # Поощряем держать выигрышные позиции и закрывать проигрышные
            if mtm > 0:
                reward += mtm * 0.1  # Меньший вес для потенциальной прибыли
            else:
                reward += mtm * 0.2  # Больший штраф для потенциальных убытков
        
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
        
        # Проверяем, не превышено ли максимальное количество шагов
        # или не достигнут ли конец данных
        done = (self.current_step >= len(self.prices) - 2) or (self.current_step >= self.max_episode_steps - 1)
        
        # Увеличиваем счетчик шагов
        self.current_step += 1
        
        # Liquidate remaining inventory at end of episode (market close)
        if done and self.inventory:
            safe_step = min(self.current_step, len(self.prices) - 1)
            final_price = float(self.prices[safe_step])
            
            # Штраф за неликвидированные позиции в конце эпизода
            # Чем больше позиций осталось, тем больше штраф
            liquidation_penalty = len(self.inventory) * 0.05
            reward -= liquidation_penalty
            
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
