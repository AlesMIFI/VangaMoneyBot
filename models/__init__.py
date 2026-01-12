"""
Импорты всех моделей
"""
from .random_forest import RandomForestModel
from .arima_model import ARIMAModel
from .prophet_model import ProphetModel
from .lstm_model import LSTMModel
from .gru_model import GRUModel

__all__ = [
    'RandomForestModel','ARIMAModel','ProphetModel','LSTMModel',
    'GRUModel'
]
