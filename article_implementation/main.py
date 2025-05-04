# -*- coding: utf-8 -*-

"""
Основной скрипт для запуска обучения и оценки TD3 агента
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from environment import TradingEnvArticle
from td3_agent import TD3Actor
from td3_agent_part2 import TD3Critic, ReplayBuffer
from td3_agent_part3 import TD3Agent
from td3_agent_part4 import train_td3_agent
from td3_agent_part5 import evaluate_td3_agent, visualize_trading_results, backtest_td3_agent
from multi_asset_wrapper import create_multi_asset_env
from technical_indicators import TechnicalIndicators

def load_data(file_path):
    """
    Загружает данные из CSV-файла.
    
    Args:
        file_path: Путь к CSV-файлу
        
    Returns:
        DataFrame с данными
    """
    print(f"Загружаем данные из {file_path}...")
    df = pd.read_csv(file_path)
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"В файле отсутствует столбец {col}")
    
    # Преобразуем дату в datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Сортируем по дате
    df = df.sort_values('Date')
    
    # Сбрасываем индекс
    df = df.reset_index(drop=True)
    
    print(f"Загружено {len(df)} строк данных")
    return df

def prepare_data(df, train_ratio=0.8):
    """
    Подготавливает данные для обучения и тестирования.
    
    Args:
        df: DataFrame с данными
        train_ratio: Доля данных для обучения
        
    Returns:
        Кортеж (train_data, test_data)
    """
    # Разделяем данные на обучающую и тестовую выборки
    train_size = int(len(df) * train_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"Размер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")
    
    return train_data, test_data

def load_news_data(news_path):
    """
    Загружает новостные данные.
    
    Args:
        news_path: Путь к файлу с новостями
        
    Returns:
        DataFrame с новостями
    """
    print(f"Загружаем новостные данные из {news_path}...")
    news_df = pd.read_csv(news_path)
    
    # Преобразуем дату в datetime
    if 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date'])
    
    print(f"Загружено {len(news_df)} новостей")
    return news_df

def main():
    """
    Основная функция для запуска обучения и оценки TD3 агента.
    """
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description='TD3 Agent for Trading')
    parser.add_argument('--train_data', type=str, help='Path to training data CSV file or directory with multiple files')
    parser.add_argument('--test_data', type=str, help='Path to test data CSV file or directory with multiple files')
    parser.add_argument('--window_size', type=int, default=47, help='Window size for analysis')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--max_inventory', type=int, default=8, help='Maximum inventory')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='Policy noise')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='Noise clip')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='Exploration noise')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--load_model', type=str, help='Path to load model from')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, no training')
    parser.add_argument('--use_sentiment', action='store_true', help='Use sentiment analysis')
    parser.add_argument('--news_data', type=str, help='Path to news data CSV file')
    parser.add_argument('--multi_asset', action='store_true', help='Use multi-asset environment')
    parser.add_argument('--report_freq', type=int, default=1000, help='Частота промежуточных отчетов (шагов)')
    args = parser.parse_args()
    
    # Отключаем sentiment_analysis если не установлен transformers
    args.use_sentiment = False
    
    # Создаем директорию для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"td3_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем данные
    data_dict = {}
    
    if args.train_data:
        if os.path.isdir(args.train_data):
            # Если указана директория, загружаем все CSV файлы
            csv_files = glob.glob(os.path.join(args.train_data, "*.csv"))
            for file_path in csv_files:
                ticker = os.path.basename(file_path).split('.')[0]
                data_dict[ticker] = load_data(file_path)
        else:
            # Если указан один файл
            ticker = os.path.basename(args.train_data).split('.')[0]
            data_dict[ticker] = load_data(args.train_data)
    else:
        # Если не указаны данные, загружаем все CSV файлы из директории data
        csv_files = [
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\GOOG.csv",
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\HPQ.csv",
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\IBM.csv",
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\MSFT.csv",
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\NFLX.csv",
            "c:\\Users\\10.ADMIN\\Documents\\GitHub\\Deep-Trade-Q\\data\\TSLA.csv"
        ]
        for file_path in csv_files:
            if os.path.exists(file_path):
                ticker = os.path.basename(file_path).split('.')[0]
                data_dict[ticker] = load_data(file_path)
    
    # Проверяем длины всех датафреймов и удаляем несовпадающие
    min_len = min([len(df) for df in data_dict.values()])
    filtered = {k: v for k, v in data_dict.items() if len(v) == min_len}
    removed = [k for k in data_dict if k not in filtered]
    if removed:
        print(f"[WARN] Исключены тикеры с несовпадающей длиной: {removed}")
    data_dict = filtered
    if len(data_dict) < 2:
        raise ValueError("Недостаточно тикеров с одинаковой длиной для мультиактивной среды!")
    
    # Удаляем тикеры с неправильной длиной данных
    lengths = [len(df) for df in data_dict.values()]
    mode_len = max(set(lengths), key=lengths.count)
    to_del = [ticker for ticker, df in data_dict.items() if len(df) != mode_len]
    if to_del:
        print(f"[WARN] Удаляю тикеры с неправильной длиной: {to_del}")
        for ticker in to_del:
            del data_dict[ticker]
    # Повторная проверка
    lengths = [len(df) for df in data_dict.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"DataFrames всё ещё разной длины! lengths={lengths}")
    
    # Проверяем, что данные загружены
    if not data_dict:
        raise ValueError("Не удалось загрузить данные. Укажите путь к данным.")
    
    # Проверяем валидность и длину DataFrame
    for ticker in list(data_dict.keys()):
        if len(data_dict[ticker]) == 0 or not isinstance(data_dict[ticker], pd.DataFrame):
            raise ValueError(f"DataFrame для {ticker} пустой или невалидный!")
    
    # Разделяем данные на обучающие и тестовые
    train_dict = {}
    test_dict = {}
    
    for ticker, df in data_dict.items():
        train_df, test_df = prepare_data(df)
        train_dict[ticker] = train_df
        test_dict[ticker] = test_df
    
    # Загружаем и обрабатываем новостные данные, если указаны
    sentiment_data = None
    if args.use_sentiment and args.news_data:
        news_df = load_news_data(args.news_data)
        
        # Инициализируем анализатор настроений
        # sentiment_analyzer = SentimentAnalyzer()
        # news_integrator = NewsSentimentIntegrator(sentiment_analyzer)
        
        # Обрабатываем новости для каждого тикера
        # sentiment_data = {}
        # for ticker in data_dict.keys():
        #     sentiment_features = news_integrator.process_news_data(news_df, ticker)
        #     # Создаем DataFrame с настроениями
        #     dates = train_dict[ticker]['Date'].unique()
        #     sentiment_df = pd.DataFrame({
        #         'date': dates,
        #         'sentiment_score': [sentiment_features['sentiment_score']] * len(dates),
        #         'sentiment_magnitude': [sentiment_features['sentiment_magnitude']] * len(dates)
        #     })
        #     sentiment_df.set_index('date', inplace=True)
        #     sentiment_data[ticker] = sentiment_df
    
    # Рассчитываем технические индикаторы для всех данных
    for ticker in data_dict.keys():
        train_dict[ticker] = TechnicalIndicators.calculate_all_indicators(train_dict[ticker])
        test_dict[ticker] = TechnicalIndicators.calculate_all_indicators(test_dict[ticker])
    
    # Создаем среду для обучения и тестирования
    if args.multi_asset and len(data_dict) > 1:
        # Используем мультиактивную среду
        print("Используем мультиактивную торговую среду...")
        train_env = create_multi_asset_env(
            data_dict=train_dict,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            commission=args.commission,
            max_inventory=args.max_inventory,
            sentiment_data=sentiment_data if args.use_sentiment else None
        )
        
        test_env = create_multi_asset_env(
            data_dict=test_dict,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            commission=args.commission,
            max_inventory=args.max_inventory,
            sentiment_data=sentiment_data if args.use_sentiment else None
        )
    else:
        # Используем одноактивную среду
        print("Используем одноактивную торговую среду...")
        ticker = list(data_dict.keys())[0]
        train_env = TradingEnvArticle(
            df=train_dict[ticker],
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            commission=args.commission,
            max_inventory=args.max_inventory
        )
        
        test_env = TradingEnvArticle(
            df=test_dict[ticker],
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            commission=args.commission,
            max_inventory=args.max_inventory
        )
    
    # Получаем размерности пространств состояний и действий
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    max_action = float(train_env.action_space.high[0])
    
    print(f"Размерность пространства состояний: {state_dim}")
    print(f"Размерность пространства действий: {action_dim}")
    print(f"Максимальное значение действия: {max_action}")
    
    # Создаем или загружаем агента
    if args.load_model:
        print(f"Загружаем модель из {args.load_model}...")
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip
        )
        agent.load(args.load_model)
    else:
        print("Создаем нового агента...")
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip
        )
    
    # Обучаем агента, если не указан флаг eval_only
    if not args.eval_only:
        print("Начинаем обучение агента...")
        agent, callback = train_td3_agent(
            env=train_env,
            eval_env=test_env,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            batch_size=args.batch_size,
            max_timesteps=args.timesteps,
            exploration_noise=args.exploration_noise,
            log_path=output_dir,
            verbose=1,
            report_freq=args.report_freq
        )
        
        # Визуализируем результаты обучения
        callback.plot_results()
        plt.savefig(os.path.join(output_dir, "training_results.png"))
        
        # Сохраняем модель
        agent.save(os.path.join(output_dir, "final_model"))
    
    # Оцениваем агента на тестовых данных
    print("Оцениваем агента на тестовых данных...")
    eval_results = evaluate_td3_agent(
        agent=agent,
        env=test_env,
        n_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Визуализируем результаты оценки
    if not args.multi_asset:
        ticker = list(data_dict.keys())[0]
        visualize_trading_results(
            results=eval_results,
            price_data=test_dict[ticker],
            save_path=os.path.join(output_dir, "evaluation_results.png"),
            show=True
        )
        
        # Выполняем бэктест на тестовых данных
        print("Выполняем бэктест на тестовых данных...")
        backtest_results = backtest_td3_agent(
            agent=agent,
            price_data=test_dict[ticker],
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            commission=args.commission,
            verbose=1
        )
        
        # Сохраняем результаты бэктеста
        backtest_df = pd.DataFrame({
            'portfolio_value': backtest_results['portfolio_values'],
            'action': backtest_results['actions'],
            'position': backtest_results['positions']
        })
        backtest_df.to_csv(os.path.join(output_dir, "backtest_results.csv"), index=False)
    
    # Выводим итоговые метрики
    print("\nИтоговые метрики:")
    print(f"Коэффициент Шарпа: {eval_results['sharpe_ratio']:.2f}")
    print(f"Максимальная просадка: {eval_results['mean_max_drawdown']:.2%}")
    if not args.multi_asset:
        print(f"Доходность: {(backtest_results['final_portfolio_value'] - args.initial_balance) / args.initial_balance:.2%}")
        print(f"Количество сделок: {backtest_results['buy_trades'] + backtest_results['sell_trades']}")
    
    print(f"\nВсе результаты сохранены в директории: {output_dir}")

if __name__ == "__main__":
    main()
