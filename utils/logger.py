"""
Логирование запросов пользователей
"""
import csv
import os
from datetime import datetime
from config import LOG_FILE
import logging

logger = logging.getLogger(__name__)


def init_log_file():
    """Инициализация файла логов с заголовками"""
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'user_id',
                'timestamp',
                'ticker',
                'amount',
                'best_model',
                'rmse',
                'mape',
                'predicted_profit',
                'recommendation'
            ])
        logger.info(f"Log file initialized: {LOG_FILE}")


def log_analysis(
    user_id: int,
    ticker: str,
    amount: float,
    best_model: str,
    rmse: float,
    mape: float,
    predicted_profit: float,
    recommendation: str = "N/A"
):
    """
    Логирование одного запроса анализа
    
    Args:
        user_id: ID пользователя Telegram
        ticker: Тикер акции
        amount: Сумма инвестиции
        best_model: Название лучшей модели
        rmse: Метрика RMSE
        mape: Метрика MAPE
        predicted_profit: Предсказанная прибыль
        recommendation: Рекомендация (buy/sell/hold)
    """
    init_log_file()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                user_id,
                timestamp,
                ticker,
                amount,
                best_model,
                f"{rmse:.4f}",
                f"{mape:.2f}",
                f"{predicted_profit:.2f}",
                recommendation
            ])
        
        logger.info(f"Logged analysis for user {user_id}: {ticker}")
        
    except Exception as e:
        logger.error(f"Failed to log analysis: {e}")
