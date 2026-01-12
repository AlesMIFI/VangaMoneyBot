"""
ARIMA модель для прогнозирования временных рядов
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from models.base_model import BaseModel
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """ARIMA (AutoRegressive Integrated Moving Average) модель"""
    
    def __init__(self, order: tuple = (5, 1, 0)):
        """
        Инициализация ARIMA модели
        
        Args:
            order: (p, d, q) параметры ARIMA
                p: порядок авторегрессии (AR)
                d: порядок дифференцирования (I)
                q: порядок скользящего среднего (MA)
        """
        super().__init__("ARIMA")
        self.order = order
        self.train_data = None
        
    def train(self, train_data: np.ndarray, **kwargs) -> None:
        """
        Обучение ARIMA
        
        Args:
            train_data: Временной ряд для обучения [samples]
        """
        logger.info(f"Training {self.name} with order {self.order}")
        
        # Проверка входных данных
        if len(train_data) < 30:
            logger.warning(f"Training data is too short: {len(train_data)} points")
        
        self.train_data = train_data
        
        try:
            # Обучение ARIMA
            arima = ARIMA(train_data, order=self.order)
            self.model = arima.fit()
            self.is_trained = True
            
            logger.info(f"{self.name} training completed")
            logger.info(f"AIC: {self.model.aic:.2f}, BIC: {self.model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error training {self.name}: {e}")
            raise
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Прогнозирование на N шагов вперед
        
        Args:
            steps: Количество шагов прогноза
            **kwargs: Дополнительные параметры (не используются)
            
        Returns:
            Массив прогнозов [steps]
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        
        try:
            # Прогноз
            forecast = self.model.forecast(steps=steps)
            
            # Конвертируем в numpy array
            forecast_array = np.array(forecast)
            
            logger.info(f"{self.name} forecast: mean={forecast_array.mean():.2f}, "
                       f"min={forecast_array.min():.2f}, max={forecast_array.max():.2f}")
            
            return forecast_array
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {e}")
            raise
    
    def get_residuals(self) -> np.ndarray:
        """
        Получение остатков модели
        
        Returns:
            Массив остатков
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        return self.model.resid
    
    def summary(self) -> str:
        """
        Получение статистики модели
        
        Returns:
            Строка со статистикой
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        return str(self.model.summary())

