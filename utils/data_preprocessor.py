"""
Предобработка данных для моделей временных рядов 
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и валидация временного ряда
    
    1. Удаление дубликатов дат
    2. Сортировка по времени
    3. Интерполяция пропусков
    4. Удаление аномалий (price <= 0)
    """
    df = df.copy()
    
    # Удаление дубликатов
    if df['date'].duplicated().any():
        logger.warning(f"Found {df['date'].duplicated().sum()} duplicate dates, removing...")
        df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Сортировка по дате 
    df = df.sort_values('date').reset_index(drop=True)
    
    # Интерполяция пропусков
    if df['price'].isnull().any():
        missing_count = df['price'].isnull().sum()
        logger.warning(f"Found {missing_count} missing values, interpolating...")
        df['price'] = df['price'].interpolate(method='linear')
        
        # Если первое значение NaN, заполняем backward fill
        if df['price'].iloc[0] is np.nan:
            df['price'] = df['price'].fillna(method='bfill')
    
    # Удаление некорректных цен
    if (df['price'] <= 0).any():
        invalid_count = (df['price'] <= 0).sum()
        logger.warning(f"Found {invalid_count} non-positive prices, removing...")
        df = df[df['price'] > 0].reset_index(drop=True)
    
    # Проверка на выбросы (только логирование) для информации
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df['price'] < lower_bound) | (df['price'] > upper_bound)).sum()
    if outliers > 0:
        logger.info(f"Detected {outliers} potential outliers (keeping them)")
    
    logger.info(f" Time series cleaned: {len(df)} days")
    
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.75) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделение на train/test  (temporal split)
    
    ВАЖНО: Train содержит самые СТАРЫЕ данные, Test - самые НОВЫЕ
    
    
    Args:
        df: DataFrame с колонками ['date', 'price']
        train_ratio: Доля train данных (например, 0.75 = 75%)
        
    Returns:
        train_df, test_df
    """
    split_index = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    logger.info(f"Train: {train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]} ({len(train_df)} days)")
    logger.info(f"Test:  {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]} ({len(test_df)} days)")
    
    return train_df, test_df


def create_sequences(data: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Создание последовательностей для одношагового прогноза
    
    Аргументы:
        data: Временной ряд [samples, 1]
        n_steps: Длина входной последовательности (look-back window)
        
    Возвращает:
        X: [samples, n_steps] - входные последовательности
        y: [samples] - целевые значения (следующая точка)
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    
    return np.array(X), np.array(y)


def prepare_data_for_ml(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lag_features: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Подготовка данных для Random Forest (без data leakage)
    
    Аргументы:
        train_df: Train DataFrame
        test_df: Test DataFrame
        lag_features: Количество лаговых признаков

    Возвращает:
        X_train, X_test, y_train, y_test
    """
    train_prices = train_df['price'].values
    test_prices = test_df['price'].values
    
    # Создание лагов для train (не видит test)
    X_train, y_train = create_sequences(train_prices, lag_features)
    
    # Для test используем overlap с train
    # Нужно lag_features последних точек train + весь test
    overlap = train_prices[-lag_features:]
    combined_test = np.concatenate([overlap, test_prices])
    
    X_test, y_test = create_sequences(combined_test, lag_features)
    
    logger.info(f"ML data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def prepare_data_for_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_steps: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Подготовка данных для LSTM/GRU (без data leakage)
    
    Ключевые принципы:
    1. Scaler обучается ТОЛЬКО на train данных
    2. Train и test создаются отдельно
    3. Используется temporal overlap для test
    
    Аргументы:
        train_df: Train DataFrame
        test_df: Test DataFrame
        n_steps: Длина входной последовательности (look-back)
        
    Возвращает:
        X_train, X_test, y_train, y_test, scaler
    """
    train_prices = train_df['price'].values.reshape(-1, 1)
    test_prices = test_df['price'].values.reshape(-1, 1)
    
    # 1. Создаем и обучаем scaler ТОЛЬКО на train
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices)  # fit ТОЛЬКО на train!
    
    logger.info(f"Scaler fitted on train: min={scaler.data_min_[0]:.2f}, max={scaler.data_max_[0]:.2f}")
    
    # 2. Масштабируем test используя параметры train
    test_scaled = scaler.transform(test_prices)  # transform (БЕЗ fit!)
    
    # 3. Создаем последовательности для train (не видит test!)
    X_train, y_train = create_sequences(train_scaled, n_steps)
    
    # 4. Для test используем overlap: последние n_steps точек train + весь test
    overlap = train_scaled[-n_steps:]
    combined_test = np.vstack([overlap, test_scaled])
    
    X_test, y_test = create_sequences(combined_test, n_steps)
    
    # 5. Reshape для LSTM: [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    logger.info(f"LSTM data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


def prepare_full_data_for_forecast(
    df: pd.DataFrame,
    model_type: str,
    n_steps: int = 30
):
    """
    Подготовка ВСЕХ данных для финального прогноза
    
    После выбора лучшей модели переобучаем на ВСЕХ данных
    
    Args:
        df: Полный DataFrame
        model_type: 'ml' или 'lstm'
        n_steps: Длина последовательности
        
    Returns:
        Для ML: X_full, y_full
        Для LSTM: X_full, y_full, scaler, last_sequence
    """
    prices = df['price'].values
    
    if model_type == 'ml':
        # Random Forest - простые лаги
        X_full, y_full = create_sequences(prices, n_steps)
        logger.info(f"Full ML data: X={X_full.shape}, y={y_full.shape}")
        return X_full, y_full
        
    elif model_type == 'lstm':
        # LSTM/GRU - масштабирование + последовательности
        prices_reshaped = prices.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(prices_reshaped)
        
        X_full, y_full = create_sequences(scaled, n_steps)
        X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
        
        # Последняя последовательность для прогноза
        last_sequence = scaled[-n_steps:].flatten()
        
        logger.info(f"Full LSTM data: X={X_full.shape}, last_seq={last_sequence.shape}")
        
        return X_full, y_full, scaler, last_sequence
