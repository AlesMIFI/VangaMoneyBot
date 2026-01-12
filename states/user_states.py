"""
Состояния пользователя для FSM
"""
from aiogram.fsm.state import State, StatesGroup


class AnalysisStates(StatesGroup):
    """Состояния процесса анализа акций"""
    waiting_for_ticker = State()   # Ожидание ввода тикера
    waiting_for_amount = State()   # Ожидание ввода суммы
    confirmation = State()          # Подтверждение данных
    processing = State()            # Обработка запроса
