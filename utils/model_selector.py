"""
Модуль для обучения, сравнения и выбора лучшей модели - ФИНАЛЬНАЯ ВЕРСИЯ
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
import logging

from models.random_forest import RandomForestModel
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from utils.data_preprocessor import (
    split_train_test,
    prepare_data_for_ml,
    prepare_data_for_lstm
)
from config import TRAIN_TEST_SPLIT, FORECAST_DAYS

logger = logging.getLogger(__name__)


class ModelSelector:
    """Класс для выбора лучшей модели прогнозирования"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация
        
        Args:
            df: DataFrame с колонками ['date', 'price']
        """
        self.df = df
        self.train_df = None
        self.test_df = None
        self.models = []
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        
    def prepare_data(self):
        """Разделение данных на train и test"""
        logger.info("Preparing data for training...")
        self.train_df, self.test_df = split_train_test(self.df, TRAIN_TEST_SPLIT)
        logger.info(f"Train: {len(self.train_df)} days, Test: {len(self.test_df)} days")
        
    def train_all_models(self) -> Dict[str, Dict]:
        """
        Обучение всех 5 моделей и оценка на тестовых данных
        
        Returns:
            Словарь с результатами каждой модели
        """
        logger.info("Starting training of all models...")
        
        train_prices = self.train_df['price'].values
        test_prices = self.test_df['price'].values
        train_dates = self.train_df['date']
        
        # ==================== 1. Random Forest ====================
        logger.info("=" * 50)
        logger.info("Training Random Forest...")
        try:
            rf_result = self._train_random_forest(train_prices, test_prices)
            self.results['Random Forest'] = rf_result
        except Exception as e:
            import traceback
            logger.error(f"Random Forest failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.results['Random Forest'] = None
        
        # ==================== 2. ARIMA ====================
        logger.info("=" * 50)
        logger.info("Training ARIMA...")
        try:
            arima_result = self._train_arima(train_prices, test_prices)
            self.results['ARIMA'] = arima_result
        except Exception as e:
            import traceback
            logger.error(f"ARIMA failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.results['ARIMA'] = None
        
        # ==================== 3. Prophet ====================
        logger.info("=" * 50)
        logger.info("Training Prophet...")
        try:
            prophet_result = self._train_prophet(train_prices, test_prices, train_dates)
            self.results['Prophet'] = prophet_result
        except Exception as e:
            import traceback
            logger.error(f"Prophet failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.results['Prophet'] = None
        
        # ==================== 4. LSTM ====================
        logger.info("=" * 50)
        logger.info("Training LSTM...")
        try:
            lstm_result = self._train_lstm(train_prices, test_prices)  # ← ИСПОЛЬЗУЕМ МЕТОД!
            self.results['LSTM'] = lstm_result
        except Exception as e:
            import traceback
            logger.error(f"LSTM failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.results['LSTM'] = None
        
        # ==================== 5. GRU ====================
        logger.info("=" * 50)
        logger.info("Training GRU...")
        try:
            gru_result = self._train_gru(train_prices, test_prices)
            self.results['GRU'] = gru_result
        except Exception as e:
            import traceback
            logger.error(f"GRU failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.results['GRU'] = None
        
        logger.info("=" * 50)
        logger.info("All models trained!")
        
        return self.results
    
    def _train_random_forest(self, train_prices: np.ndarray, test_prices: np.ndarray) -> Dict:
        """Обучение Random Forest"""
        n_lags = 10
        
        # Подготовка данных (передаем DataFrame!)
        X_train, X_test, y_train, y_test = prepare_data_for_ml(
            self.train_df,
            self.test_df,
            lag_features=n_lags
        )
        
        # Обучение
        model = RandomForestModel(n_estimators=100, n_lags=n_lags)
        model.train(X_train, y_train)
        
        # Оценка на test
        metrics = model.evaluate_with_features(X_test, y_test)
        
        return {
            'model': model,
            'metrics': metrics,
            'test_predictions': None,
            'n_lags': n_lags,
            'last_values': train_prices[-n_lags:]
        }
    
    def _train_arima(self, train_prices: np.ndarray, test_prices: np.ndarray) -> Dict:
        """Обучение ARIMA"""
        model = ARIMAModel(order=(5, 1, 0))
        model.train(train_prices)
        
        # Прогноз на тестовых данных
        predictions = model.predict(len(test_prices))
        
        # Оценка метрик
        metrics = model.evaluate(test_prices, predictions)
        
        return {
            'model': model,
            'metrics': metrics,
            'test_predictions': predictions
        }
    
    def _train_prophet(self, train_prices: np.ndarray, test_prices: np.ndarray, train_dates: pd.Series) -> Dict:
        """Обучение Prophet"""
        model = ProphetModel()
        
        # ВАЖНО: dates ПЕРВЫЕ!
        model.train(train_dates, train_prices)
        
        # Генерируем даты для test
        test_dates = self.test_df['date']
        
        # Прогноз на КОНКРЕТНЫЕ даты test
        predictions = model.predict(future_dates=test_dates)
        
        # Оценка метрик
        metrics = model.evaluate(test_prices, predictions)
        
        return {
            'model': model,
            'metrics': metrics,
            'test_predictions': predictions
        }
    
    def _train_lstm(self, train_prices: np.ndarray, test_prices: np.ndarray) -> Dict:
        """Обучение LSTM"""
        n_steps = 30
        
        # Подготовка данных (передаем DataFrame!)
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(
            self.train_df,  # ← DataFrame!
            self.test_df,   # ← DataFrame!
            n_steps=n_steps
        )
        
        # Обучение
        model = LSTMModel(n_steps=n_steps)
        model.train(X_train, y_train, scaler)
        
        # Оценка на test
        metrics = model.evaluate_with_features(X_test, y_test)  # ← evaluate_with_features!
        
        # Для финального прогноза
        full_data = np.concatenate([train_prices, test_prices])
        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        full_scaled = temp_scaler.fit_transform(full_data.reshape(-1, 1)).flatten()
        last_sequence_full = full_scaled[-n_steps:]
        
        return {
            'model': model,
            'metrics': metrics,
            'test_predictions': None,
            'last_sequence': last_sequence_full,
            'scaler': temp_scaler
        }
    
    def _train_gru(self, train_prices: np.ndarray, test_prices: np.ndarray) -> Dict:
        """Обучение GRU"""
        n_steps = 30
        
        # Подготовка данных (передаем DataFrame!)
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(
            self.train_df,  # ← DataFrame!
            self.test_df,   # ← DataFrame!
            n_steps=n_steps
        )
        
        # Обучение
        model = GRUModel(n_steps=n_steps)
        model.train(X_train, y_train, scaler)
        
        # Оценка на test
        metrics = model.evaluate_with_features(X_test, y_test)  # ← evaluate_with_features!
        
        # Для финального прогноза
        full_data = np.concatenate([train_prices, test_prices])
        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        full_scaled = temp_scaler.fit_transform(full_data.reshape(-1, 1)).flatten()
        last_sequence_full = full_scaled[-n_steps:]
        
        return {
            'model': model,
            'metrics': metrics,
            'test_predictions': None,
            'last_sequence': last_sequence_full,
            'scaler': temp_scaler
        }
    
    def select_best_model(self) -> Tuple[str, Dict, Dict]:
        """
        Выбор лучшей модели по RMSE
        
        Returns:
            (model_name, model_info, metrics)
        """
        logger.info("Selecting best model...")
        
        # Фильтруем успешно обученные модели
        valid_results = {name: result for name, result in self.results.items() if result is not None}
        
        if not valid_results:
            raise ValueError("No models were successfully trained!")
        
        # Выбираем модель с минимальным RMSE
        best_name = min(valid_results.keys(), key=lambda x: valid_results[x]['metrics']['rmse'])
        best_result = valid_results[best_name]
        
        self.best_model_name = best_name
        self.best_model = best_result['model']
        self.best_metrics = best_result['metrics']
        
        logger.info(f"Best model: {best_name}")
        logger.info(f"Metrics: RMSE={self.best_metrics['rmse']:.4f}, MAPE={self.best_metrics['mape']:.2f}%")
        
        return best_name, best_result, self.best_metrics
    
    def retrain_and_forecast(self) -> np.ndarray:
        """
        Переобучение лучшей модели на ВСЕХ данных и прогноз на 30 дней
        
        Returns:
            Массив прогнозов [30]
        """
        logger.info(f"Retraining {self.best_model_name} on full dataset...")
        
        all_prices = self.df['price'].values
        all_dates = self.df['date']
        
        # Переобучаем на всех данных
        if self.best_model_name == 'Random Forest':
            result = self.results['Random Forest']
            n_lags = result['n_lags']
            
            # Создаем лаги на всех данных
            X_all, y_all = self._create_lags(all_prices, n_lags)
            
            # Переобучаем
            new_model = RandomForestModel(n_estimators=100, n_lags=n_lags)
            new_model.train(X_all, y_all)
            
            # Прогноз на 30 дней
            last_values = all_prices[-n_lags:]
            forecast = new_model.predict(FORECAST_DAYS, last_values)
            
        elif self.best_model_name == 'ARIMA':
            new_model = ARIMAModel(order=(5, 1, 0))
            new_model.train(all_prices)
            forecast = new_model.predict(FORECAST_DAYS)
            
        elif self.best_model_name == 'Prophet':
            new_model = ProphetModel()
            new_model.train(all_dates, all_prices)  # ← dates ПЕРВЫЕ!
            forecast = new_model.predict(steps=FORECAST_DAYS)  # ← steps (не future_dates)
            
        elif self.best_model_name == 'LSTM':
            result = self.results['LSTM']
            forecast = result['model'].predict(FORECAST_DAYS, result['last_sequence'])
            
        elif self.best_model_name == 'GRU':
            result = self.results['GRU']
            forecast = result['model'].predict(FORECAST_DAYS, result['last_sequence'])
        
        logger.info(f"Forecast generated for {FORECAST_DAYS} days")
        
        return forecast
    
    def _create_lags(self, data: np.ndarray, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
        """Вспомогательная функция для создания лагов"""
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i-n_lags:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Создание таблицы сравнения всех моделей
        
        Returns:
            DataFrame с результатами
        """
        data = []
        
        for name, result in self.results.items():
            if result is not None:
                metrics = result['metrics']
                data.append({
                    'Модель': name,
                    'RMSE': f"{metrics['rmse']:.4f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'MAPE': f"{metrics['mape']:.2f}%",
                    'Лучшая': '✅' if name == self.best_model_name else ''
                })
            else:
                data.append({
                    'Модель': name,
                    'RMSE': 'ERROR',
                    'MAE': 'ERROR',
                    'MAPE': 'ERROR',
                    'Лучшая': ''
                })
        
        df = pd.DataFrame(data)
        return df
