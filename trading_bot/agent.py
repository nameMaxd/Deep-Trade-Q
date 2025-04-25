import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from trading_bot.utils import WINDOW_SIZE


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning"""
    error = y_true - y_pred
    # condition for small error
    cond = tf.abs(error) <= clip_delta
    # squared loss for small errors
    squared_loss = 0.5 * tf.square(error)
    # linear loss for large errors
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    # combine
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=5000, pretrained=False, model_name=None, buy_threshold=-0.01):
        self.strategy = strategy

        # agent config
        # Используем переданный window_size для расчёта размера состояния
        self.window_size = WINDOW_SIZE
        self.state_size = WINDOW_SIZE - 1 + 4    # теперь включает vol_ratio
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.buy_threshold = 0.0  # Lower threshold for actions

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # Soften epsilon decay to keep exploration longer
        self.epsilon_decay = 0.999  # Slower epsilon decay
        # Use a lower learning rate for stability
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        # Use gradient clipping to stabilize training
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.tau = 2.0  # Higher temperature for exploration

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Создает модель: Dense или LSTM (выбор через self.model_type)"""
        from keras.layers import Dropout, BatchNormalization, LSTM, Dense, Input, Reshape
        from keras.regularizers import l2
        from keras.models import Sequential
        if hasattr(self, 'model_type') and self.model_type == 'lstm':
            # LSTM-архитектура, shape (timesteps, features)
            model = Sequential()
            model.add(Reshape((self.state_size, 1), input_shape=(self.state_size,)))
            model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(256, return_sequences=False, kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(self.action_size))
            model.compile(loss=self.loss, optimizer=self.optimizer)
            return model
        else:
            # Dense-архитектура (по умолчанию)
            model = Sequential()
            model.add(Dense(units=128, activation="relu", input_dim=self.state_size, kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(units=128, activation="relu", kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(units=self.action_size))
            model.compile(loss=self.loss, optimizer=self.optimizer)
            return model


    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False, episode=0):
        """Take action (Boltzmann sampling in training, threshold-greedy in eval)"""
        if not is_eval:
            epsilon = max(0.5, 1.0 - (episode / 1000))  # Min 50% random actions
            if random.random() < epsilon:
                return random.choice([0, 1, 2])
        q_values = self.model.predict(state, verbose=0)[0]
        # Boltzmann sampling during training
        if not is_eval:
            logits = q_values / self.tau
            exp_q = np.exp(logits - np.max(logits))
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(self.action_size, p=probs)
        # Greedy threshold decision during evaluation
        if q_values[1] - q_values[0] > self.buy_threshold:
            return 1
        if q_values[2] - q_values[0] > self.buy_threshold and self.inventory:
            return 2
        return 0

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state, verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state, verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][np.argmax(self.model.predict(next_state, verbose=0)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state, verbose=0)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
        filename = f"models/{self.model_name}_{episode}.h5"
        self.model.save(filename)

    def load(self):
        import logging
        import os
        from keras.models import load_model
        # Исправлено: model_path теперь всегда корректный
        model_path = self.model_name if os.path.isabs(self.model_name) else os.path.join("models", self.model_name)
        if model_path.endswith(".h5") and os.path.exists(model_path):
            model = self._model()
            try:
                model.load_weights(model_path)
                logging.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logging.warning(f"Could not load weights from {model_path}: {e}. Training from scratch.")
            return model
        else:
            try:
                return load_model(model_path, custom_objects=self.custom_objects)
            except Exception as e:
                logging.warning(f"Could not load model from {model_path}: {e}. Training from scratch.")
                return self._model()
