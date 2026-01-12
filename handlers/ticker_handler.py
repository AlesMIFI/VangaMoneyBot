"""
Обработчик ввода тикера акции
"""
from aiogram import Router, F
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramNetworkError
import logging

from states.user_states import AnalysisStates
from keyboards.main_keyboards import get_cancel_keyboard, get_main_menu
from utils.validators import validate_ticker
from utils.data_loader import check_ticker_exists

router = Router()
logger = logging.getLogger(__name__)


@router.message(F.text == "Начать анализ")
async def start_analysis(message: Message, state: FSMContext):
    """Начало процесса анализа"""
    await state.set_state(AnalysisStates.waiting_for_ticker)
    
    await message.answer(
        "Введите <b>тикер акции</b> (например: AAPL, MSFT, TSLA, GOOGL):\n\n"
        "Тикер — это короткое обозначение компании на бирже.",
        reply_markup=get_cancel_keyboard(),
        parse_mode="HTML"
    )


@router.message(AnalysisStates.waiting_for_ticker, F.text == "❌ Отменить")
async def cancel_ticker_input(message: Message, state: FSMContext):
    """Отмена ввода"""
    await state.clear()
    await message.answer(
        "❌Анализ отменён.",
        reply_markup=get_main_menu()
    )


@router.message(AnalysisStates.waiting_for_ticker)
async def process_ticker(message: Message, state: FSMContext):
    """Обработка введённого тикера"""
    ticker = message.text.strip().upper()
    
    try:
        # 1. Базовая валидация формата
        is_valid, error_message = validate_ticker(ticker)
        
        if not is_valid:
            await message.answer(
                f"❌ {error_message}\n\n"
                "Попробуйте снова или нажмите <b>❌ Отменить</b>.",
                reply_markup=get_cancel_keyboard(),
                parse_mode="HTML"
            )
            return
        
        # 2. Отправка сообщения о проверке
        checking_msg = await message.answer(
            f"🔍 Проверяю тикер <b>{ticker}</b>...",
            parse_mode="HTML"
        )
        
        # 3. Асинхронная проверка существования тикера
        exists, error_msg = await check_ticker_exists(ticker)
        
        # Удаляем сообщение о проверке
        try:
            await checking_msg.delete()
        except:
            pass
        
        if not exists:
            await message.answer(
                f"❌ {error_msg}\n\n"
                "Попробуйте снова или нажмите <b>❌ Отменить</b>.",
                reply_markup=get_cancel_keyboard(),
                parse_mode="HTML"
            )
            return
        
        # 4. Тикер валидный и существует
        await state.update_data(ticker=ticker)
        await state.set_state(AnalysisStates.waiting_for_amount)
        
        await message.answer(
            f"✅ Тикер <b>{ticker}</b> найден!\n\n"
            f"Теперь введите <b>сумму инвестиции</b> в долларах (целое число):\n"
            f"Например: 10000",
            reply_markup=get_cancel_keyboard(),
            parse_mode="HTML"
        )
        
    except TelegramNetworkError as e:
        logger.error(f"Telegram Network Error для пользователя {message.from_user.id}: {e}")
        await state.clear()
        try:
            await message.answer(
                "❌Произошла сетевая ошибка. Попробуйте позже.",
                reply_markup=get_main_menu()
            )
        except:
            pass
            
    except Exception as e:
        logger.error(f"Неожиданная ошибка в process_ticker: {e}")
        await state.clear()
        try:
            await message.answer(
                "❌Произошла ошибка. Попробуйте начать заново.",
                reply_markup=get_main_menu()
            )
        except:
            pass

