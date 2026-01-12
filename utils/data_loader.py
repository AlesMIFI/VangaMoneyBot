"""
Загрузка данных акций с Yahoo Finance (асинхронная версия)
необходимо так как yfinance не поддерживает асинхронность из коробки в отличие aiogram
"""
import yfinance as yf
import pandas as pd
from typing import Tuple, Optional
from config import DATA_PERIOD
import logging
import asyncio

logger = logging.getLogger(__name__)


def _check_ticker_sync(ticker: str) -> Tuple[bool, str]:
    """
    Синхронная проверка тикера (вызывается из async функции)
    """
    try:
        # Позволяем yfinance самому управлять сессией
        stock = yf.Ticker(ticker)
        
        # Быстрая проверка - только 5 последних дней
        hist = stock.history(period="5d")
        
        if hist.empty:
            return False, (
                f"Тикер <b>{ticker}</b> не найден или нет данных.\n\n"
                f"Проверьте написание. Популярные тикеры:\n"
                f"   • AAPL (Apple)\n"
                f"   • MSFT (Microsoft)\n"
                f"   • GOOGL (Google)\n"
                f"   • TSLA (Tesla)\n"
                f"   • AMZN (Amazon)"
            )
        
        logger.info(f"Ticker {ticker} validated successfully, got {len(hist)} days")
        return True, ""
        
    except Exception as e:  # Общий перехват исключений для логирования (например, сетевые ошибки)
        logger.error(f"Ошибка проверки тикера {ticker}: {e}")
        return False, (
            f"Не удалось проверить тикер <b>{ticker}</b>.\n\n"
            f"Возможные причины:\n"
            f"   • Проблемы с подключением к Yahoo Finance\n"
            f"   • Неверный тикер\n\n"
            f"Попробуйте позже или другой тикер."
        )


async def check_ticker_exists(ticker: str) -> Tuple[bool, str]:
    """
    Асинхронная проверка существования тикера
    
    Аргументы:
        ticker(Тикер акции)
        
    Возвращает:
        (exists, error_message)
    """
    try:
        # Выполняем синхронную функцию в отдельном потоке
        exists, error_msg = await asyncio.to_thread(_check_ticker_sync, ticker)
        return exists, error_msg
    except asyncio.TimeoutError:
        return False, "Превышено время ожидания ответа от Yahoo Finance."
    except Exception as e:
        logger.error(f"Неожиданная ошибка при проверке {ticker}: {e}")
        return False, f"Произошла ошибка при проверке тикера."


def _load_stock_data_sync(ticker: str, period: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Синхронная загрузка данных (вызывается из async функции)
    """
    try:
        logger.info(f"Загрузка данных для {ticker} за период {period}")
        
        # Позволяем yfinance самому управлять сессией
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return None, f"Не удалось загрузить данные для {ticker}."
        
        # Проверка минимального количества данных
        if len(df) < 100:
            return None, (
                f"Недостаточно исторических данных для <b>{ticker}</b>.\n"
                f"Требуется минимум 100 дней, доступно: {len(df)} дней."
            )
        
        # Оставляем только цену закрытия
        df = df[['Close']].copy()
        df.columns = ['price']
        
        # Сброс индекса
        df = df.reset_index()
        df.columns = ['date', 'price']
        
        logger.info(f"Загружено {len(df)} записей для {ticker}")
        
        return df, ""
        
    except Exception as e:
        logger.error(f"Ошибка загрузки данных для {ticker}: {e}")
        return None, f"Ошибка загрузки данных: {str(e)}"


async def load_stock_data(ticker: str, period: str = DATA_PERIOD) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Асинхронная загрузка исторических данных акций
    
    Аргументы:
        ticker(Тикер акции)
        period(Период данных (по умолчанию 2 года)) # Используется константа из config.py

    Возвращает:
        (dataframe, error_message)
    """
    try:
        # Выполняем синхронную функцию в отдельном потоке
        df, error_msg = await asyncio.to_thread(_load_stock_data_sync, ticker, period)
        return df, error_msg
    except asyncio.TimeoutError:
        return None, "Превышено время ожидания загрузки данных."
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке {ticker}: {e}")
        return None, "Произошла ошибка при загрузке данных."


def _get_stock_info_sync(ticker: str) -> dict:
    """
    Синхронное получение информации о компании
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': info.get('currentPrice', 0)
        }
    except Exception as e:
        logger.error(f"Ошибка получения информации для {ticker}: {e}")
        return {
            'name': ticker,
            'sector': 'N/A',
            'current_price': 0
        }


async def get_stock_info(ticker: str) -> dict:
    """
    Асинхронное получение информации о компании

    Аргументы:
        ticker(Тикер акции)

    Возвращает:
        Словарь с информацией
    """
    return await asyncio.to_thread(_get_stock_info_sync, ticker)


