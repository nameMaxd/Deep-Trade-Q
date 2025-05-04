"""
Реализация анализа настроений для торговой системы
согласно статье "Deep Reinforcement Learning Approach for Trading Automation in The Stock Market"
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

class SentimentAnalyzer:
    """
    Анализатор настроений для финансовых новостей, как описано в статье.
    Использует FinBERT для анализа настроений.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Инициализирует анализатор настроений.
        
        Args:
            model_name: Имя предобученной модели
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Загружаем модель {model_name} для анализа настроений...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Классы настроений
        self.labels = ["negative", "neutral", "positive"]
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Анализирует текст и возвращает оценки настроений.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с оценками настроений
        """
        # Токенизируем текст
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем предсказания
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Преобразуем логиты в вероятности
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        
        # Формируем результат
        result = {label: float(prob) for label, prob in zip(self.labels, probabilities)}
        
        return result
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Анализирует пакет текстов и возвращает оценки настроений для каждого.
        
        Args:
            texts: Список текстов для анализа
            batch_size: Размер пакета
            
        Returns:
            Список словарей с оценками настроений
        """
        results = []
        
        # Обрабатываем тексты пакетами
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+batch_size]
            
            # Токенизируем тексты
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Получаем предсказания
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Преобразуем логиты в вероятности
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Формируем результаты
            batch_results = [
                {label: float(prob) for label, prob in zip(self.labels, probs)}
                for probs in probabilities
            ]
            
            results.extend(batch_results)
        
        return results
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Возвращает числовую оценку настроения текста.
        Положительное значение означает позитивное настроение,
        отрицательное - негативное, близкое к нулю - нейтральное.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Числовая оценка настроения
        """
        sentiment = self.analyze_text(text)
        
        # Вычисляем взвешенную оценку
        score = sentiment["positive"] - sentiment["negative"]
        
        return score

class NewsSentimentIntegrator:
    """
    Интегратор новостных настроений в торговую систему, как описано в статье.
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Инициализирует интегратор новостных настроений.
        
        Args:
            sentiment_analyzer: Анализатор настроений
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.news_cache = {}  # Кэш для хранения проанализированных новостей
    
    def process_news_data(self, news_df: pd.DataFrame, ticker: str) -> Dict[str, float]:
        """
        Обрабатывает новостные данные для заданного тикера.
        
        Args:
            news_df: DataFrame с новостями
            ticker: Тикер акции
            
        Returns:
            Словарь с оценками настроений
        """
        # Фильтруем новости по тикеру
        ticker_news = news_df[news_df["ticker"] == ticker]
        
        if len(ticker_news) == 0:
            return {"sentiment_score": 0.0, "sentiment_magnitude": 0.0}
        
        # Получаем тексты новостей
        news_texts = ticker_news["headline"].tolist()
        
        # Анализируем настроения
        sentiment_scores = []
        for text in news_texts:
            # Проверяем, есть ли текст в кэше
            if text in self.news_cache:
                score = self.news_cache[text]
            else:
                score = self.sentiment_analyzer.get_sentiment_score(text)
                self.news_cache[text] = score
            
            sentiment_scores.append(score)
        
        # Вычисляем среднюю оценку и магнитуду
        sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        sentiment_magnitude = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_magnitude": sentiment_magnitude
        }
    
    def get_sentiment_features(self, date: str, news_df: pd.DataFrame, ticker: str) -> Dict[str, float]:
        """
        Получает признаки настроений для заданной даты и тикера.
        
        Args:
            date: Дата
            news_df: DataFrame с новостями
            ticker: Тикер акции
            
        Returns:
            Словарь с признаками настроений
        """
        # Фильтруем новости по дате
        date_news = news_df[news_df["date"] == date]
        
        if len(date_news) == 0:
            return {"sentiment_score": 0.0, "sentiment_magnitude": 0.0}
        
        # Обрабатываем новости
        sentiment_features = self.process_news_data(date_news, ticker)
        
        return sentiment_features
