import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size, commission=0.001, max_inventory=8, carry_cost=0.0001, min_trade_value=2.5, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True):
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
        self.state_size = window_size - 1 + 6
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
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_size,), dtype=np.float32
        )

    def reset(self, seed=None, options=None, random_start=False):
        super().reset(seed=seed)
        if random_start:
            self.current_step = np.random.randint(0, len(self.prices) - self.window_size - 1)
        else:
            self.current_step = 0
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
        self.hold_penalty = 0.1  # increased hold penalty to encourage trades
        # global normalization bounds if provided, else compute from data
        if self.global_min_v is not None:
            self.min_v = self.global_min_v
        else:
            self.min_v = np.min(self.prices)
        if self.global_max_v is not None:
            self.max_v = self.global_max_v
        else:
            self.max_v = np.max(self.prices)
        # get state array
        state_arr = get_state(
            list(zip(self.prices, self.volumes)),
            self.current_step, self.window_size,
            min_v=self.min_v, max_v=self.max_v
        )[0]
        return state_arr.astype(np.float32), {}

    def step(self, action):
        # map continuous to discrete action: 0=HOLD,1=BUY,2=SELL
        if isinstance(action, (list, np.ndarray)):
            a = float(action[0])
        else:
            a = float(action)
        action = int(np.clip(np.round(a), 0, 2))
        price = float(self.prices[self.current_step])
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
        # get state array
        state_arr = get_state(
            list(zip(self.prices, self.volumes)),
            self.current_step, self.window_size,
            min_v=self.min_v, max_v=self.max_v
        )[0]
        obs = state_arr.astype(np.float32)
        # info теперь содержит реальное действие
        info = {'real_action': action}
        return obs, reward, done, False, info
