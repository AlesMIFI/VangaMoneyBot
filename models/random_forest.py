"""
Random Forest модель для прогнозирования временных рядов
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest с лаговыми признаками"""
    
    def __init__(self, n_estimators: int = 100, n_lags: int = 10):
        """
        Инициализация Random Forest модели
        
        Args:
            n_estimators: Количество деревьев в лесу
            n_lags: Количество лаговых признаков (look-back window)
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.n_lags = n_lags
        self.last_values = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Обучение Random Forest
        
        Args:
            X_train: Признаки (лаговые значения) [samples, n_lags]
            y_train: Целевая переменная [samples]
        """
        logger.info(f"Training {self.name} with {self.n_estimators} estimators")
        
        # Проверка размерности
        if X_train.shape[1] != self.n_lags:
            logger.warning(f"Expected {self.n_lags} lags, got {X_train.shape[1]}")
            self.n_lags = X_train.shape[1]
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Feature importance (опционально)
        feature_importance = self.model.feature_importances_
        logger.info(f"{self.name} training completed")
        logger.info(f"Top 3 important lags: {np.argsort(feature_importance)[-3:][::-1]}")
    
    def predict(self, steps: int, last_values: np.ndarray, **kwargs) -> np.ndarray:
        """
        Многошаговое прогнозирование (итеративное)
        
        Args:
            steps: Количество шагов прогноза
            last_values: Последние n_lags значений [n_lags]
            **kwargs: Дополнительные параметры
            
        Returns:
            Массив прогнозов [steps]
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        if len(last_values) != self.n_lags:
            raise ValueError(f"last_values must have length {self.n_lags}, got {len(last_values)}")
        
        predictions = []
        current_input = last_values.copy()
        
        for step in range(steps):
            # Прогноз следующего значения
            pred = self.model.predict(current_input.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Обновление входных данных (сдвиг окна)
            # [t-9, t-8, ..., t-1, t] → [t-8, t-7, ..., t, pred]
            current_input = np.append(current_input[1:], pred)
        
        predictions = np.array(predictions)
        
        logger.info(f"{self.name} forecast: mean=${predictions.mean():.2f}, "
                   f"min=${predictions.min():.2f}, max=${predictions.max():.2f}")
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Получение важности признаков (лагов)
        
        Returns:
            Массив важностей [n_lags]
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        return self.model.feature_importances_
