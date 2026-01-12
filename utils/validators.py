"""
Валидация пользовательского ввода
"""
import re
from typing import Tuple
from config import TICKER_MIN_LENGTH, TICKER_MAX_LENGTH, AMOUNT_MIN, AMOUNT_MAX


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Валидация тикера акции (базовая проверка формата)
    
    Аргументы:
        ticker: Тикер акции
        
    Возвращает:
        (is_valid, error_message)
    """
    # Проверка длины
    if len(ticker) < TICKER_MIN_LENGTH or len(ticker) > TICKER_MAX_LENGTH:
        return False, f"Тикер должен содержать от {TICKER_MIN_LENGTH} до {TICKER_MAX_LENGTH} символов."
    
    # Проверка формата (только латинские буквы)
    if not re.match(r'^[A-Z]+$', ticker):
        return False, "Тикер должен содержать только латинские буквы."
    
    return True, ""


def validate_amount(amount_str: str) -> Tuple[bool, str, int]:
    """
    Валидация суммы инвестиции
    
    Аргументы:
        amount_str: Строка с суммой
        
    Возвращает:
        (is_valid, error_message, amount)
    """
    # Проверка на пустую строку
    if not amount_str:
        return False, "Сумма не может быть пустой.", 0
    
    # Проверка на целое число
    try:
        amount = int(amount_str)
    except ValueError:
        return False, "Сумма должна быть целым числом (без букв, точек и запятых).", 0
    
    # Проверка диапазона
    if amount < AMOUNT_MIN:
        return False, f"Сумма должна быть больше ${AMOUNT_MIN}.", 0
    
    if amount > AMOUNT_MAX:
        return False, f"Сумма не должна превышать ${AMOUNT_MAX:,}.", 0
    
    return True, "", amount

