import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size=47, commission=0.001, min_trade_value=1000.0, max_inventory=8, carry_cost=0.0001, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True, stop_loss_pct=0.05):
        super().__init__()
        self.prices = prices
        self.volumes = volumes
        self.window_size = window_size
        # features: window_size-1 sigmoids + SMA, EMA, RSI, vol_ratio + momentum, volatility
        self.state_size = window_size - 1 + 6  # без +2, inventory добавляется отдельно!
        # transaction commission fraction
        self.commission = commission
        # minimum trade value threshold (in $) to open a position
        self.min_trade_value = min_trade_value  # Увеличиваем размер сделки для более заметного профита
        # position limits and holding costs
        self.max_inventory = max_inventory  # Максимальное количество позиций ограничено до 8
        self.carry_cost = carry_cost
        # global normalization bounds override
        self.global_min_v = min_v
        self.global_max_v = max_v
        # risk aversion parameters
        self.risk_lambda = risk_lambda
        self.drawdown_lambda = drawdown_lambda
        self.dual_phase = dual_phase
        # Параметры стоп-лоссов
        self.stop_loss_pct = stop_loss_pct  # Процент стоп-лосса (5% по умолчанию)
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
        
        # Улучшенная система вознаграждений для более точного следования тренду
        # Анализируем тренд цены за последние 10 шагов для более стабильного определения тренда
        trend_window_short = 5  # Короткий тренд (5 шагов)
        trend_window_long = 10   # Длинный тренд (10 шагов)
        
        # Короткий тренд (для быстрых решений)
        start_idx_short = max(0, self.current_step - trend_window_short)
        price_window_short = self.prices[start_idx_short:self.current_step + 1]
        
        # Длинный тренд (для стратегических решений)
        start_idx_long = max(0, self.current_step - trend_window_long)
        price_window_long = self.prices[start_idx_long:self.current_step + 1]
        
        # Определяем направление короткого тренда
        if len(price_window_short) > 1:
            trend_short = np.polyfit(np.arange(len(price_window_short)), price_window_short, 1)[0]
            trend_strength_short = abs(trend_short) / np.mean(price_window_short) * 100
        else:
            trend_short = 0
            trend_strength_short = 0
        
        # Определяем направление длинного тренда
        if len(price_window_long) > 1:
            trend_long = np.polyfit(np.arange(len(price_window_long)), price_window_long, 1)[0]
            trend_strength_long = abs(trend_long) / np.mean(price_window_long) * 100
        else:
            trend_long = 0
            trend_strength_long = 0
        
        # Проверяем, совпадают ли направления трендов (более надежный сигнал)
        trends_aligned = (trend_short > 0 and trend_long > 0) or (trend_short < 0 and trend_long < 0)
        
        # Базовые вознаграждения за действия
        if action == 1:  # Покупка
            # Базовое вознаграждение за активную торговлю
            step_reward = 0.5
            
            # Вознаграждение за покупку в соответствии с трендом
            if trend_long > 0:  # Восходящий долгосрочный тренд - хорошо для покупки
                # Больше вознаграждение, если оба тренда совпадают
                if trends_aligned:
                    step_reward += 2.0 + min(trend_strength_long, 10.0) / 5.0
                else:
                    step_reward += 1.0 + min(trend_strength_long, 10.0) / 10.0
            else:  # Нисходящий долгосрочный тренд - плохо для покупки
                # Меньше штраф, если короткий тренд положительный (возможный разворот)
                if trend_short > 0:
                    step_reward -= 0.3
                else:
                    step_reward -= 1.0 + min(trend_strength_long, 10.0) / 20.0
            
        elif action == 2:  # Продажа
            # Базовое вознаграждение за активную торговлю
            step_reward = 0.5
            
            # Вознаграждение за продажу в соответствии с трендом
            if trend_long < 0:  # Нисходящий долгосрочный тренд - хорошо для продажи
                # Больше вознаграждение, если оба тренда совпадают
                if trends_aligned:
                    step_reward += 2.0 + min(trend_strength_long, 10.0) / 5.0
                else:
                    step_reward += 1.0 + min(trend_strength_long, 10.0) / 10.0
            else:  # Восходящий долгосрочный тренд - плохо для продажи
                # Меньше штраф, если короткий тренд отрицательный (возможный разворот)
                if trend_short < 0:
                    step_reward -= 0.3
                else:
                    step_reward -= 1.0 + min(trend_strength_long, 10.0) / 20.0
            
        else:  # Держать
            # Штраф за бездействие, но меньше если нет чёткого тренда
            if abs(trend_long) < 0.001 or not trends_aligned:
                step_reward = -0.5  # Меньший штраф, если тренд неясен
            else:
                step_reward = -1.0  # Больший штраф при явном тренде
            
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

        # Проверяем стоп-лоссы перед выполнением действия агента
        # Это позволяет защитить от больших убытков независимо от решения агента
        stop_loss_triggered = False
        stop_loss_reward = 0
        
        # Проходим по всем позициям и проверяем стоп-лоссы
        inventory_with_stop_loss = []
        for i, (bought_price, qty, position_size) in enumerate(self.inventory):
            # Рассчитываем текущий убыток в процентах
            current_loss_pct = (bought_price - price) / bought_price
            
            # Если убыток превышает порог стоп-лосса, закрываем позицию
            if current_loss_pct >= self.stop_loss_pct:
                stop_loss_triggered = True
                
                # Рассчитываем прибыль/убыток от закрытия позиции по стоп-лоссу
                profit = (price - bought_price) * qty
                cost = self.commission * (price * qty + bought_price * qty)
                net = profit - cost
                
                # Добавляем к награде и общей прибыли
                stop_loss_reward += net
                self.total_profit += net
                
                # Увеличиваем счетчик сделок
                self.trade_count += 1
            else:
                # Если стоп-лосс не сработал, оставляем позицию в инвентаре
                inventory_with_stop_loss.append((bought_price, qty, position_size))
        
        # Обновляем инвентарь после проверки стоп-лоссов
        self.inventory = inventory_with_stop_loss
        
        # Если сработал хотя бы один стоп-лосс, добавляем соответствующую награду
        if stop_loss_triggered:
            self.last_action_step = self.current_step
        
        # Получаем вознаграждение на основе действия и изменения цены
        reward = self._calculate_reward(action) + stop_loss_reward

        # Обрабатываем действия агента
        if action == 0:  # HOLD
            # Штраф за удержание уже учтен в _calculate_reward
            pass
            
        elif action == 1:  # BUY
            # Динамическое управление размером позиции
            if len(self.inventory) < self.max_inventory:
                # Определяем размер позиции в зависимости от уверенности в тренде
                # Используем короткий и длинный тренды для определения размера позиции
                trend_window_short = 5
                trend_window_long = 10
                
                # Короткий тренд (для быстрых решений)
                start_idx_short = max(0, self.current_step - trend_window_short)
                price_window_short = self.prices[start_idx_short:self.current_step + 1]
                
                # Длинный тренд (для стратегических решений)
                start_idx_long = max(0, self.current_step - trend_window_long)
                price_window_long = self.prices[start_idx_long:self.current_step + 1]
                
                # Определяем направление трендов
                if len(price_window_short) > 1 and len(price_window_long) > 1:
                    trend_short = np.polyfit(np.arange(len(price_window_short)), price_window_short, 1)[0]
                    trend_long = np.polyfit(np.arange(len(price_window_long)), price_window_long, 1)[0]
                    
                    # Проверяем, совпадают ли направления трендов (более надежный сигнал)
                    trends_aligned = (trend_short > 0 and trend_long > 0)
                    
                    # Базовый размер позиции
                    base_position_size = 25.0  # Минимальный размер позиции $25
                    
                    # Определяем множитель размера позиции в зависимости от силы тренда
                    if trends_aligned and trend_short > 0 and trend_long > 0:
                        # Сильный восходящий тренд - увеличиваем размер позиции
                        trend_strength = abs(trend_long) / np.mean(price_window_long) * 100
                        position_multiplier = min(4.0, 1.0 + trend_strength / 5.0)  # Максимум 4x от базового размера
                    else:
                        # Слабый или противоречивый тренд - используем базовый размер
                        position_multiplier = 1.0
                    
                    # Рассчитываем итоговый размер позиции, но не более 1000/8 = 125$ на позицию
                    position_size = min(125.0, base_position_size * position_multiplier)
                else:
                    # Если недостаточно данных, используем базовый размер
                    position_size = 25.0
                
                # Покупаем акции на рассчитанную сумму
                qty = position_size / price
                self.inventory.append((price, qty, position_size))  # Сохраняем также размер позиции в долларах
                
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
                # Извлекаем первую позицию из инвентаря (FIFO)
                bought_price, qty, position_size = self.inventory.pop(0)
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
                # Штраф за попытку продать без позиций
                reward -= 1.0 * self.hold_penalty
        
        # carry cost per item in inventory
        reward -= self.carry_cost * len(self.inventory)
        
        # update state
        self.current_step += 1
        
        # Calculate MTM P&L for observation
        next_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        mtm = sum((next_price - price) * qty for price, qty, _ in self.inventory)
        
        # Calculate inventory value for observation
        inventory_value = sum(qty for _, qty, _ in self.inventory)
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
            for bought_price, qty, position_size in self.inventory:
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
            total_qty = sum(qty for _, qty, _ in self.inventory)
            avg_price = sum(bp * qty for bp, qty, _ in self.inventory) / total_qty
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
