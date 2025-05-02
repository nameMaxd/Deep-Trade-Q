import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, prices, volumes, window_size, commission=0.001, max_inventory=8, carry_cost=0.0001, min_trade_value=10.0, min_v=None, max_v=None, risk_lambda=0.1, drawdown_lambda=0.1, dual_phase=True, max_episode_steps=None, hold_penalty=0.01, profit_bonus=0.2, loss_penalty=0.1, long_hold_penalty_factor=0.0001, trade_frequency_bonus=0.05, max_drawdown_penalty=0.5, normalize_prices=True):
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
        self.long_hold_penalty_factor = long_hold_penalty_factor
        self.trade_frequency_bonus = trade_frequency_bonus
        self.max_drawdown_penalty = max_drawdown_penalty
        self.normalize_prices = normalize_prices
        
        # Нормализация цен для устранения разрыва между историческими и текущими ценами
        if self.normalize_prices:
            # Сохраняем оригинальные цены
            self.original_prices = self.prices.copy()
            # Нормализуем цены относительно их среднего и стандартного отклонения
            # Это сохраняет относительные движения, но устраняет абсолютную разницу
            mean_price = np.mean(self.prices)
            std_price = np.std(self.prices)
            if std_price > 0:
                self.prices = (self.prices - mean_price) / std_price * 10 + 100  # Центрируем около 100
            else:
                self.prices = np.ones_like(self.prices) * 100
        
        # Счетчики для отслеживания торговой активности
        self.trade_count = 0
        self.hold_duration = {}  # Словарь для отслеживания длительности удержания каждой позиции

    def reset(self, *args, seed=None, options=None, random_start=False, **kwargs):
        # Важно: возвращаем кортеж (observation, info)
        if seed is not None:
            np.random.seed(seed)
        
        # Сбрасываем состояние среды
        self.inventory = []
        self.total_profit = 0.0
        self.rewards = []
        self.equity = [0.0]  # Начальный капитал
        self.max_equity = 0.0
        
        # Сбрасываем счетчики торговой активности
        self.trade_count = 0
        self.hold_duration = {}
        
        # Устанавливаем начальный шаг
        if random_start and len(self.prices) > self.window_size + 100:
            # Случайное начало для разнообразия данных
            self.current_step = np.random.randint(self.window_size, len(self.prices) - 100)
        else:
            # Начинаем с первого возможного шага после окна
            self.current_step = self.window_size
        
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
        
        # Получаем текущую цену (нормализованную)
        price = float(self.prices[self.current_step])
        
        # Получаем оригинальную цену для логов
        if hasattr(self, 'original_prices') and self.normalize_prices:
            original_price = float(self.original_prices[self.current_step])
        else:
            original_price = price
        
        reward = 0.0
        
        # Обновляем счетчики длительности удержания для всех позиций
        for i, (bp, qty) in enumerate(self.inventory):
            if i not in self.hold_duration:
                self.hold_duration[i] = 0
            self.hold_duration[i] += 1
        
        # Проверяем, есть ли позиции, которые держатся слишком долго
        # Если есть, принудительно продаем самую старую
        force_sell = False
        if self.inventory:
            oldest_position_idx = 0
            oldest_duration = self.hold_duration.get(0, 0)
            
            # Если позиция держится более 10 дней, принудительно продаем
            if oldest_duration > 10:
                force_sell = True
                print(f"ПРИНУДИТЕЛЬНАЯ ПРОДАЖА: позиция держится {oldest_duration} дней")
                # Записываем в лог-файл тоже
                with open("force_sell.log", "a", encoding="utf-8") as f:
                    f.write(f"ПРИНУДИТЕЛЬНАЯ ПРОДАЖА: позиция держится {oldest_duration} дней\n")
                action = 2  # SELL
        
        # РАДИКАЛЬНОЕ ИЗМЕНЕНИЕ: Если у нас уже есть позиции, с вероятностью 50% заставляем агента продавать
        if self.inventory and np.random.random() < 0.5:
            original_action = action
            action = 2  # SELL
            with open("force_sell.log", "a", encoding="utf-8") as f:
                f.write(f"РАДИКАЛЬНОЕ ИЗМЕНЕНИЕ: Заставляем агента продавать (было {original_action}, стало {action})\n")
        
        # Применяем действие
        if action == 1:  # BUY
            # Проверяем, не превышен ли лимит позиций
            if len(self.inventory) < self.max_inventory:
                # Проверяем, достаточно ли ценность сделки
                if original_price >= self.min_trade_value:
                    # Покупаем 1 акцию по текущей цене
                    self.inventory.append((price, 1.0))
                    # Комиссия как штраф
                    commission_cost = self.commission * price
                    reward -= commission_cost
                    
                    # Увеличиваем счетчик сделок
                    self.trade_count += 1
                    
                    # Добавляем новую запись в словарь длительности удержания
                    self.hold_duration[len(self.inventory) - 1] = 0
                    
                    # Бонус за активную торговлю
                    reward += self.trade_frequency_bonus
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
                
                # Базовая награда - чистая прибыль
                reward += net_profit
                
                # Длительность удержания этой позиции
                hold_time = self.hold_duration.pop(0)
                
                # Сдвигаем индексы в словаре длительности удержания
                new_hold_duration = {}
                for i, duration in self.hold_duration.items():
                    new_hold_duration[i-1] = duration
                self.hold_duration = new_hold_duration
                
                # Добавляем бонус за прибыльную сделку или штраф за убыточную
                if profit > 0:
                    # Бонус за прибыльную сделку
                    profit_bonus = profit * self.profit_bonus
                    
                    # Дополнительный бонус за быструю прибыльную сделку
                    if hold_time < 10:  # Если держали меньше 10 дней
                        profit_bonus *= 1.5  # Увеличиваем бонус на 50%
                    
                    reward += profit_bonus
                else:
                    # Штраф за убыточную сделку
                    loss_penalty = abs(profit) * self.loss_penalty
                    
                    # Меньший штраф, если быстро закрыли убыточную позицию
                    if hold_time < 5:  # Если закрыли убыток быстро
                        loss_penalty *= 0.5  # Уменьшаем штраф на 50%
                    
                    reward -= loss_penalty
                
                # Увеличиваем счетчик сделок
                self.trade_count += 1
                
                # Бонус за активную торговлю
                reward += self.trade_frequency_bonus
                
                # Дополнительный бонус за продажу (чтобы стимулировать продажи)
                sell_bonus = self.trade_frequency_bonus * 5.0  # Увеличиваем бонус за продажу в 5 раз
                reward += sell_bonus
                
                # Логируем бонус для отладки
                print(f"БОНУС ЗА ПРОДАЖУ: +{sell_bonus:.2f}")
                
                self.total_profit += net_profit
            else:
                # Нечего продавать, не продаем
                action = 0  # HOLD
                # Штраф за попытку продать без позиций
                reward -= 0.01
        
        else:  # HOLD
            # Применяем штраф за удержание позиций (для предотвращения стратегии "купи и держи")
            if self.inventory:
                # Базовый штраф за удержание
                hold_penalty = self.hold_penalty * len(self.inventory)
                
                # Дополнительный штраф, растущий со временем удержания
                for i, duration in self.hold_duration.items():
                    # Экспоненциальный рост штрафа с увеличением длительности удержания
                    # Чем дольше держим позицию, тем больше штраф
                    # Используем квадратичную функцию для более быстрого роста штрафа
                    exponential_penalty = self.long_hold_penalty_factor * (duration ** 2)
                    hold_penalty += exponential_penalty
                
                # Применяем штраф
                reward -= hold_penalty
                
                # Добавляем дополнительный штраф, если агент держит позиции до конца эпизода
                if self.current_step >= len(self.prices) - 10:  # Если мы близко к концу эпизода
                    # Драконовский штраф за удержание позиций до конца эпизода
                    end_of_episode_penalty = len(self.inventory) * 2.0  # 200% штраф за каждую позицию
                    reward -= end_of_episode_penalty
                    
                    # Логируем штраф для отладки
                    print(f"ШТРАФ ЗА УДЕРЖАНИЕ ДО КОНЦА: -{end_of_episode_penalty:.2f} (позиций: {len(self.inventory)})")
            
            # Базовый штраф за удержание (бездействие)
            reward -= self.hold_penalty
        
        # Штраф за удержание позиций (стоимость капитала)
        # Штраф растет с увеличением длительности удержания
        if self.inventory:
            total_holding_penalty = 0.0
            
            for i, (bp, qty) in enumerate(self.inventory):
                # Базовая стоимость удержания
                holding_cost = self.carry_cost * bp * qty
                
                # Длительность удержания этой позиции
                hold_time = self.hold_duration.get(i, 0)
                
                # Штраф растет квадратично с длительностью удержания
                # Это сильно наказывает за стратегию "купи и держи"
                time_factor = 1.0 + (hold_time * self.long_hold_penalty_factor) ** 2
                
                # Итоговый штраф за удержание этой позиции
                position_penalty = holding_cost * time_factor
                total_holding_penalty += position_penalty
            
            reward -= total_holding_penalty
        
        # Оценка потенциальной прибыли/убытка (mark-to-market)
        if self.current_step < len(self.prices) - 1 and self.inventory:
            next_price = float(self.prices[min(self.current_step + 1, len(self.prices) - 1)])
            mtm = sum((next_price - price) * qty for price, qty in self.inventory)
            
            # Поощряем держать выигрышные позиции и закрывать проигрышные
            if mtm > 0:
                reward += mtm * 0.1  # Меньший вес для потенциальной прибыли
            else:
                reward += mtm * 0.2  # Больший штраф для потенциальных убытков
        
        # Обновляем метрики риска
        self.rewards.append(reward)
        self.equity.append(self.equity[-1] + reward)
        self.max_equity = max(self.max_equity, self.equity[-1])
        
        # Расчет просадки (drawdown)
        current_drawdown = (self.max_equity - self.equity[-1]) / (self.max_equity + 1e-8)
        
        # Штраф за просадку
        if current_drawdown > 0.05:  # Если просадка больше 5%
            drawdown_penalty = current_drawdown * self.max_drawdown_penalty
            reward -= drawdown_penalty
        
        # Расчет Sharpe Ratio на основе последних доходностей
        returns = self.rewards[-20:] if len(self.rewards) > 20 else self.rewards
        
        if len(returns) > 1:
            # Волатильность доходности (риск)
            vol = float(np.std(returns))
            # Средняя доходность
            mean_return = float(np.mean(returns))
            
            # Штраф за высокий риск (волатильность)
            risk_penalty = self.risk_lambda * vol
            reward -= risk_penalty
            
            # Бонус за стабильную положительную доходность
            if mean_return > 0 and vol < mean_return:
                reward += mean_return * 0.1  # Бонус за хороший риск/доходность
        
        # Проверяем, не превышено ли максимальное количество шагов
        # или не достигнут ли конец данных
        done = (self.current_step >= len(self.prices) - 2)
        if self.max_episode_steps is not None:
            done = done or (self.current_step >= self.max_episode_steps - 1)
        
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
                
                # Штраф за стратегию "купи и держи" до конца эпизода
                if self.current_step > 100:  # Если эпизод достаточно длинный
                    reward -= abs(net) * 0.2  # Штраф 20% от прибыли/убытка
            
            self.inventory = []
            self.hold_duration = {}
        
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
