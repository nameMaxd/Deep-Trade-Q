import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .ops import get_state
from config import *  # Импортируем все параметры

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def _calculate_volume_profile(self, prices, volumes):
        """Рассчитывает профиль объема на основе параметров из config.py"""
        self.volume_profile = []
        self.support_resistance_levels = []
        
        for i in range(len(prices)):
            start_idx = max(0, i - VOLUME_LOOKBACK)
            end_idx = i + 1
            
            if end_idx - start_idx < 10:
                price_range = np.linspace(min(prices[:end_idx]), max(prices[:end_idx]), VOLUME_BINS)
                vol_profile = np.zeros(VOLUME_BINS)
            else:
                price_range = np.linspace(min(prices[start_idx:end_idx]), max(prices[start_idx:end_idx]), VOLUME_BINS)
                vol_profile = np.zeros(VOLUME_BINS)
                
                for j in range(start_idx, end_idx):
                    bin_idx = np.abs(price_range - prices[j]).argmin()
                    vol_profile[bin_idx] += volumes[j]
            
            if np.sum(vol_profile) > 0:
                vol_profile = vol_profile / np.sum(vol_profile)
            
            support_resistance = price_range[np.argsort(vol_profile)[-3:]]
            
            self.volume_profile.append(vol_profile)
            self.support_resistance_levels.append(support_resistance)
            
        self.volume_profile = np.array(self.volume_profile)
        self.support_resistance_levels = np.array(self.support_resistance_levels)

    def __init__(self, prices, volumes, window_size=WINDOW_SIZE, commission=COMMISSION, 
                 min_trade_value=MIN_TRADE_VALUE, max_inventory=MAX_INVENTORY, 
                 carry_cost=CARRY_COST, min_v=None, max_v=None, risk_lambda=RISK_LAMBDA, 
                 drawdown_lambda=DRAWDOWN_LAMBDA, dual_phase=DUAL_PHASE, stop_loss_pct=STOP_LOSS_PCT):
        super().__init__()
        self.prices = prices
        self.volumes = volumes
        self.window_size = window_size
        # features: window_size-1 sigmoids + SMA, EMA, RSI, vol_ratio + momentum, volatility
        self.state_size = window_size - 1 + 6  # без +2, inventory добавляется отдельно!
        # transaction commission fraction
        self.commission = commission
        # minimum trade value threshold (in $) to open a position
        self.min_trade_value = min_trade_value
        # position limits and holding costs
        self.max_inventory = max_inventory
        self.carry_cost = carry_cost
        # global normalization bounds override
        self.global_min_v = min_v
        self.global_max_v = max_v
        # risk aversion parameters
        self.risk_lambda = risk_lambda
        self.drawdown_lambda = drawdown_lambda
        # dual-phase training flag
        self.dual_phase = dual_phase
        # stop-loss percentage
        self.stop_loss_pct = stop_loss_pct
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
        # Сброс состояния среды
        self.inventory = []
        self.equity = [0.0]
        self.max_equity = 0.0
        self.rewards = []
        self.total_profit = 0.0
        self.trade_count = 0
        self.win_count = 0  # Счетчик выигрышных сделок
        self.inaction_counter = 0  # Счетчик бездействия
        # Инициализация фазы
        self.phase = 'exploration' if self.dual_phase else 'exploitation'
        self.current_step = 0
        self.last_action_step = 0
        self.hold_penalty = 0.0  # Adaptive penalty for holding
        

        
        # Normalize price data
        self.min_v = np.min(self.prices) if self.global_min_v is None else self.global_min_v
        self.max_v = np.max(self.prices) if self.global_max_v is None else self.global_max_v
        
        # Вычисляем профиль горизонтального объема, если ещё не вычислен
        if not hasattr(self, 'volume_profile') or self.volume_profile is None:
            self._calculate_volume_profile(self.prices, self.volumes)
        
        # Get initial state
        # Используем старый метод получения состояния
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

    # def _calculate_reward(self, action):
    #     """
    #     Новый TD3-стиль: награда = изменение equity с учетом комиссии и штрафа за незначительные сделки.
    #     """
    #     # Текущий equity: сумма денег + стоимость всех позиций по текущей цене
    #     current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
    #     # Считаем стоимость инвентаря
    #     inventory_value = sum([qty * current_price for _, qty, _ in self.inventory])
    #     equity = getattr(self, 'cash', 0.0) + inventory_value
    #     # Предыдущий equity (на прошлом шаге)
    #     if not hasattr(self, '_last_equity'):
    #         self._last_equity = equity
    #     # Изменение equity
    #     delta_equity = equity - self._last_equity
    #     # Усиливаем мотивацию к крупным сделкам и штрафуем за копеечные сделки
    #     commission = self.commission * max(abs(delta_equity), self.min_trade_value)
    #     small_trade_penalty = 0.01
    #     # Если сделка меньше min_trade_value — жестко штрафуем
    #     if abs(delta_equity) < self.min_trade_value:
    #         small_trade_penalty = -2.0 * self.commission * self.min_trade_value
    #     self._last_equity = equity
    #     # Итоговая награда: изменение equity - комиссия - штраф
    #     reward = delta_equity - commission + small_trade_penalty
    #     return reward

    def _calculate_reward(self, action):
    # Текущая цена и стоимость позиций
        current_price = self.prices[self.current_step]
        
        # Реализованная прибыль от закрытых позиций
        realized_pnl = sum([self._calculate_closed_pnl(price, qty, entry_price) 
                            for price, qty, entry_price in self.closed_positions])
        
        # Нереализованная прибыль открытых позиций
        unrealized_pnl = sum([self._calculate_unrealized_pnl(current_price, qty, entry_price) 
                            for _, qty, entry_price in self.open_positions])
        
        # Комиссии по всем сделкам
        total_commission = sum([abs(price * qty) * self.commission 
                            for price, qty, _ in self.all_trades])
        
        # Проскальзывание (пример простой модели)
        slippage_penalty = sum([abs(price - current_price) * abs(qty) 
                            for price, qty, _ in self.last_step_trades])
        
        # Основная награда: реализованная прибыль - комиссии - проскальзывание
        base_reward = realized_pnl - total_commission - slippage_penalty
        
        # Штраф за держание позиции (временной риск)
        holding_penalty = -0.001 * abs(unrealized_pnl) * len(self.open_positions)
        
        # Штраф за частые сделки (overtrading)
        trade_count_penalty = -0.01 * len(self.last_step_trades)
        
        # Адаптивный штраф за маленькие сделки
        small_trade_penalty = 0
        for price, qty, _ in self.last_step_trades:
            trade_value = abs(price * qty)
            if trade_value < self.min_trade_value:
                small_trade_penalty -= (self.min_trade_value - trade_value) * 0.1
        
        # Итоговая награда
        reward = base_reward + holding_penalty + trade_count_penalty + small_trade_penalty
        
        # Сохраняем историю для анализа
        self._last_reward_components = {
            'realized_pnl': realized_pnl,
            'commissions': total_commission,
            'slippage': slippage_penalty,
            'holding_penalty': holding_penalty,
            'trade_count_penalty': trade_count_penalty,
            'small_trade_penalty': small_trade_penalty
        }
    
        return reward

    def step(self, action):
        # РАДИКАЛЬНО ИЗМЕНЕННЫЙ МЕТОД STEP С ОТЛАДКОЙ
        # Вместо использования действий модели, мы заставляем её торговать по простому алгоритму
        
        # Добавляем отладочный вывод только для первых 10 шагов, чтобы не засорять консоль
        if self.current_step < 10:
            print(f'DEBUG: Step {self.current_step}, Исходное действие: {action}, Позиций: {len(self.inventory)}, Сделок: {self.trade_count}')
        
        # Преобразуем действие из непрерывного в дискретное (но мы его всё равно заменим)
        original_action = np.clip(action, 0, 2).item()
        original_action = int(round(original_action))
        
        # Проверяем, что индекс не выходит за границы массива
        safe_step = min(self.current_step, len(self.prices) - 1)
        price = float(self.prices[safe_step])
        
        # ЗДЕСЬ МЫ ПРИНУДИТЕЛЬНО ЗАСТАВЛЯЕМ МОДЕЛЬ ТОРГОВАТЬ!
        # Простой алгоритм: покупаем, когда цена растет, продаем, когда падает
        # удалил потому что полный бред был

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
                
        elif action == 1:  # Sell
            if self.inventory:
                # Продаем самую старую позицию (FIFO)
                bought_price, qty, position_size = self.inventory.pop(0)
                profit = (price - bought_price) * qty
                cost = self.commission * (price * qty + bought_price * qty)
                net = profit - cost
                reward += net
                self.total_profit += net
                
                # Увеличиваем счетчик сделок
                self.trade_count += 1
                
                # Отслеживаем выигрышные сделки (с положительным чистым профитом)
                if net > 0:
                    self.win_count += 1
            else:
                # Строго запрещаем продажу без позиций - игнорируем действие
                action = 2  # Принудительно меняем действие на HOLD
                # Добавляем небольшой штраф за попытку продать без позиций
                reward -= 0.1  # Меньший штраф, чтобы не перевешивать реальные прибыли/убытки
        
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
            
        # РАДИКАЛЬНАЯ СИСТЕМА ВОЗНАГРАЖДЕНИЙ - ЗАСТАВЛЯЕМ ТОРГОВАТЬ!
        
        # Обновляем счетчик бездействия
        if action == 2:  # Если HOLD
            self.inaction_counter += 1
        else:  # Если покупка или продажа
            self.inaction_counter = 0  # Сбрасываем счетчик при активном действии

        # 1. МАКСИМАЛЬНЫЙ бонус за активную торговлю (если действие не HOLD)
        if action != 2:  # Если действие - покупка или продажа
            reward += 15.0  # МАКСИМАЛЬНЫЙ бонус за активность
        
        # 2. АБСОЛЮТНО НЕВЫНОСИМЫЙ штраф за бездействие, когда нет позиций
        if action == 2 and len(self.inventory) == 0:  # Если HOLD и нет позиций
            inaction_penalty = 20.0 + (self.inaction_counter * 2.0)  # Огромный штраф, очень быстро растущий с каждым шагом
            reward -= inaction_penalty  # Максимально возможный штраф за бездействие без позиций
        
        # 3. ГИГАНТСКИЙ бонус за прибыльные сделки
        if reward > 0:
            reward = reward * 10.0  # Максимально увеличиваем положительные вознаграждения
        
        # 4. Штраф за убыточные сделки делаем ПОЧТИ НУЛЕВЫМ
        if reward < 0 and action != 2:  # Если убыток от активного действия
            reward = reward * 0.05  # Практически нулевой штраф за убыточные сделки, чтобы модель не боялась экспериментировать
        
        # 5. Штраф за слишком долгое удержание позиций
        if action == 2 and len(self.inventory) > 0:  # Если действие HOLD и есть позиции
            # Добавляем штраф за удержание позиций слишком долго
            hold_time = 0
            if self.last_action_step is not None:
                hold_time = self.current_step - self.last_action_step
            
            # Прогрессивный штраф за длительное удержание позиций
            if hold_time > 3:  # Уменьшаем порог до 3 шагов
                reward -= 0.5 * hold_time  # Значительно увеличиваем штраф за удержание
        
        # 6. Убрали все бонусы за конкретные стратегии (тренд, уровни поддержки/сопротивления)
        # Это позволит модели самой найти оптимальную стратегию без навязывания конкретных паттернов
        
        # 7. МАКСИМАЛЬНЫЙ бонус за частоту сделок
        if self.trade_count > 0 and self.current_step > 0:
            trades_per_step = self.trade_count / (self.current_step + 1)
            reward += trades_per_step * 50.0  # Огромный бонус за высокую частоту сделок
            
            # Дополнительный бонус за каждую сделку
            reward += self.trade_count * 0.1  # Бонус растет с каждой сделкой
        
        # 8. Ограничиваем вознаграждение, но делаем диапазон ОЧЕНЬ БОЛЬШИМ
        reward = np.clip(reward, -50.0, 50.0)  # Максимально расширяем диапазон для сверхсильных сигналов
        
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
            
        # Проверяем, завершен ли эпизод
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
        
        # Примечание: мы не добавляем дополнительные фичи горизонтального объема в состояние,
        # но используем их для вычисления вознаграждений
        
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
        
        # БОЛЕЕ РАЗУМНАЯ ТОРГОВАЯ СТРАТЕГИЯ
        # Перезаписываем действие в самом конце метода, но только если модель слишком долго не торгует
        
        # Считаем количество шагов с последней сделки
        steps_since_last_action = self.current_step - self.last_action_step
        
        # Если модель не торговала более 20 шагов, заставляем её торговать
        if steps_since_last_action > 20 and action == 2:  # Только если модель выбрала HOLD
            # Проверяем изменение цены
            if self.current_step > 0:
                prev_price = float(self.prices[max(0, safe_step - 1)])
                price_change = price - prev_price
                price_change_pct = price_change / prev_price if prev_price > 0 else 0
                
                # Если цена значительно растет (>0.5%) - покупаем
                if price_change_pct > 0.005:
                    action = 0  # BUY
                # Если цена значительно падает (>0.5%) и есть позиции - продаем
                elif price_change_pct < -0.005 and len(self.inventory) > 0:
                    action = 1  # SELL
                # Если цена сильно падает (>1%) и нет позиций - покупаем на дне
                elif price_change_pct < -0.01 and len(self.inventory) == 0:
                    # С вероятностью 50% покупаем на сильном падении (покупка на дне)
                    if np.random.random() < 0.5:
                        action = 0  # BUY
        
        # Если модель не торговала более 50 шагов, принудительно заставляем сделать случайное действие
        if steps_since_last_action > 50:
            # Случайно выбираем между покупкой и продажей
            if len(self.inventory) > 0:
                # Если есть позиции, то с вероятностью 70% продаем
                action = 1 if np.random.random() < 0.7 else 0
            else:
                # Если нет позиций, то покупаем
                action = 0
        
        # ТЕПЕРЬ НАМ НУЖНО ЗАНОВО ВЫПОЛНИТЬ ДЕЙСТВИЕ!
        # Реализуем действие вручную, минуя все проверки
        
        # Если действие - покупка
        if action == 0:
            # Рассчитываем количество акций, которое можем купить
            max_affordable = self.min_trade_value / price
            
            # Добавляем в инвентарь
            self.inventory.append((price, max_affordable, self.min_trade_value))
            
            # Увеличиваем счетчик сделок
            self.trade_count += 1
            
            # Сбрасываем счетчик бездействия
            self.inaction_counter = 0
            
            # Запоминаем шаг последнего действия
            self.last_action_step = self.current_step
        
        # Если действие - продажа и есть позиции
        elif action == 1 and len(self.inventory) > 0:
            # Продаем самую старую позицию (FIFO)
            bought_price, qty, position_size = self.inventory.pop(0)
            
            # Рассчитываем прибыль/убыток
            profit = (price - bought_price) * qty
            cost = self.commission * (price * qty + bought_price * qty)
            net = profit - cost
            
            # Добавляем к общей прибыли
            self.total_profit += net
            
            # Увеличиваем счетчик сделок
            self.trade_count += 1
            
            # Если сделка прибыльная, увеличиваем счетчик выигрышей
            if net > 0:
                self.win_count += 1
            
            # Сбрасываем счетчик бездействия
            self.inaction_counter = 0
            
            # Запоминаем шаг последнего действия
            self.last_action_step = self.current_step
        
        # info теперь содержит реальное действие
        info = {'real_action': action}
        
        # Добавляем отладочный вывод в конце метода
        if self.current_step < 10:
            print(f'DEBUG END: Step {self.current_step}, Финальное действие: {action}, Позиций: {len(self.inventory)}, Сделок: {self.trade_count}, Награда: {reward}')
        
        return obs, reward, done, False, info
