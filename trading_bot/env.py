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
        self.min_trade_value = min_trade_value
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
        # choose episode phase
        if self.dual_phase:
            self.phase = 'exploration' if np.random.rand() < 0.5 else 'exploitation'
        else:
            self.phase = 'exploitation'
        # penalty for holding to force actions
        self.hold_penalty = 0.01  # reduced hold penalty to discourage unnecessary holds
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

    def step(self, action):
        # map continuous to discrete action: 0=HOLD,1=BUY,2=SELL
        if isinstance(action, (list, np.ndarray)):
            a = float(action[0])
        else:
            a = float(action)
        action = int(np.clip(np.round(a), 0, 2))
        
        # Проверяем, что индекс не выходит за границы массива
        safe_step = min(self.current_step, len(self.prices) - 1)
        price = float(self.prices[safe_step])
        
        # initialize reward: penalize hold, apply commission on buy/sell, enforce limits
        reward = 0.0
        if action == 0:
            reward -= self.hold_penalty
        elif action == 1:
            # BUY: invest fixed amount self.min_trade_value (fractional share)
            if len(self.inventory) < self.max_inventory:
                qty = self.min_trade_value / price
                self.inventory.append((price, qty))
                reward -= self.commission * price * qty
            else:
                reward -= self.hold_penalty
        elif action == 2:
            # SELL
            if self.inventory:
                bought_price, qty = self.inventory.pop(0)
                profit = (price - bought_price) * qty
                cost = self.commission * (price * qty + bought_price * qty)
                net = profit - cost
                reward += net
                self.total_profit += net
            else:
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
