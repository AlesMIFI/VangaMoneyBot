"""
Базовый класс для всех моделей прогнозирования
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Базовый класс для моделей"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, train_data: np.ndarray, **kwargs) -> None:
        """
        Обучение модели
        
        Args:
            train_data: Обучающие данные
            **kwargs: Дополнительные параметры
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Прогнозирование
        
        Args:
            steps: Количество шагов для прогноза
            **kwargs: Дополнительные параметры
            
        Returns:
            Массив прогнозных значений
        """
        pass
    
    def evaluate(self, y_test: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Оценка модели (для ARIMA, Prophet - когда predictions УЖЕ есть)
        
        Args:
            y_test: Истинные значения
            predictions: Предсказанные значения
            
        Returns:
            Словарь с метриками
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        # ВАЖНО: y_test должен быть numpy array, не pandas Series!
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Убеждаемся, что массивы одномерные
        if len(y_test.shape) > 1:
            y_test = y_test.flatten()
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Вычисление метрик
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        logger.info(f"{self.name} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def evaluate_with_features(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Оценка модели (для Random Forest, LSTM, GRU - когда нужен X_test)
        
        Args:
            X_test: Тестовые признаки
            y_test: Истинные значения
            
        Returns:
            Словарь с метриками
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        # Получаем предсказания В ЗАВИСИМОСТИ от типа модели
        if hasattr(self, 'model') and hasattr(self.model, 'predict'):
            # Для sklearn моделей (Random Forest)
            predictions = self.model.predict(X_test)
            
        elif hasattr(self, 'device'):
            # Для PyTorch моделей (LSTM/GRU)
            import torch
            self.model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                predictions = self.model(X_tensor).cpu().numpy().flatten()
            
            # Обратное масштабирование для LSTM/GRU
            if hasattr(self, 'scaler') and self.scaler is not None:
                predictions = predictions.reshape(-1, 1)
                predictions = self.scaler.inverse_transform(predictions).flatten()
                
                y_test_reshaped = y_test.reshape(-1, 1)
                y_test = self.scaler.inverse_transform(y_test_reshaped).flatten()
        
        else:
            raise ValueError(f"Cannot determine model type for {self.name}")
        
        # ВАЖНО: y_test должен быть numpy array, не pandas Series!
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Убеждаемся, что y_test одномерный
        if len(y_test.shape) > 1:
            y_test = y_test.flatten()
        
        # Вычисление метрик
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        logger.info(f"{self.name} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
