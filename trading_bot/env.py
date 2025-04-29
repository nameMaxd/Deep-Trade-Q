import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size, commission=0.001):
        super().__init__()
        self.prices = prices
        self.volumes = volumes
        self.window_size = window_size
        # features: window_size-1 sigmoids + SMA, EMA, RSI, vol_ratio + momentum, volatility
        self.state_size = window_size - 1 + 6
        # transaction commission fraction
        self.commission = commission
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([2.0]), shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.state_size,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0.0
        # penalty for holding to force actions
        self.hold_penalty = 0.1  # increased hold penalty to encourage trades
        self.min_v = np.min(self.prices)
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
        # initialize reward: penalize hold, apply commission on buy/sell
        reward = 0.0
        if action == 0:
            reward -= self.hold_penalty
        elif action == 1:
            # BUY
            self.inventory.append(price)
            reward -= self.commission * price
        elif action == 2:
            # SELL
            if self.inventory:
                bought = self.inventory.pop(0)
                profit = price - bought
                # commission on both sides
                cost = (price + bought) * self.commission
                net = profit - cost
                reward += net
                self.total_profit += net
            else:
                reward -= self.hold_penalty
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
