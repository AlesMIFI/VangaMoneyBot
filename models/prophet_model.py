"""
Prophet модель для прогнозирования временных рядов
"""
import numpy as np
import pandas as pd
from prophet import Prophet
from models.base_model import BaseModel
import logging

# Отключаем verbose логи Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """Prophet модель от Facebook"""
    
    def __init__(self):
        super().__init__("Prophet")
        self.last_date = None
        
    def train(self, dates: pd.Series, train_data: np.ndarray, **kwargs) -> None:
        """
        Обучение Prophet
        
        Args:
            dates: Даты для временного ряда (pd.Series с datetime)
            train_data: Временной ряд значений [samples]
        """
        logger.info(f"Training {self.name}")
        
        # Проверка длины
        if len(dates) != len(train_data):
            raise ValueError(f"Dates and train_data must have the same length: {len(dates)} vs {len(train_data)}")
        
        # Prophet требует специальный формат данных: 'ds' (dates) и 'y' (values)
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates.values),  # Убедимся, что это datetime
            'y': train_data
        })
        
        # Инициализация Prophet с параметрами
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'  # Лучше для акций
        )
        
        # Обучение
        self.model.fit(df)
        
        # Сохраняем последнюю дату
        self.last_date = pd.to_datetime(dates.iloc[-1])
        self.is_trained = True
        
        logger.info(f"{self.name} training completed")
        logger.info(f"Last training date: {self.last_date}")
    
    def predict(self, steps: int = None, future_dates: pd.Series = None, **kwargs) -> np.ndarray:
        """
        Прогнозирование
        
        Args:
            steps: Количество дней для прогноза (если future_dates=None)
            future_dates: Конкретные даты для прогноза (если передано)
            **kwargs: Дополнительные параметры
            
        Returns:
            Массив прогнозов [steps] или [len(future_dates)]
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        # Вариант 1: Передали конкретные даты (для test)
        if future_dates is not None:
            df_future = pd.DataFrame({
                'ds': pd.to_datetime(future_dates.values)
            })
            
            logger.info(f"Predicting for specific dates: {df_future['ds'].iloc[0]} to {df_future['ds'].iloc[-1]}")
            
        # Вариант 2: Прогноз на N дней вперед (для финального прогноза)
        elif steps is not None:
            if steps <= 0:
                raise ValueError(f"Steps must be positive, got {steps}")
            
            df_future = self.model.make_future_dataframe(periods=steps, freq='D')
            
            logger.info(f"Predicting {steps} days into the future")
            
        else:
            raise ValueError("Either 'steps' or 'future_dates' must be provided")
        
        # Прогноз
        forecast = self.model.predict(df_future)
        
        # Извлекаем прогнозные значения
        if future_dates is not None:
            # Для test: возвращаем только прогнозы на указанные даты
            # Нужно найти их в forecast
            forecast_dates = pd.to_datetime(forecast['ds'])
            target_dates = pd.to_datetime(future_dates.values)
            
            # Находим индексы совпадающих дат
            mask = forecast_dates.isin(target_dates)
            predictions = forecast.loc[mask, 'yhat'].values
            
        else:
            # Для финального прогноза: последние N значений
            predictions = forecast['yhat'].values[-steps:]
        
        logger.info(f"{self.name} forecast: mean={predictions.mean():.2f}, "
                   f"min={predictions.min():.2f}, max={predictions.max():.2f}")
        
        return np.array(predictions)
    
    def get_forecast_components(self, steps: int) -> pd.DataFrame:
        """
        Получение компонентов прогноза (тренд, сезонность)
        
        Args:
            steps: Количество дней
            
        Returns:
            DataFrame с компонентами
        """
        if not self.is_trained:
            raise ValueError(f"{self.name} is not trained yet")
        
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
