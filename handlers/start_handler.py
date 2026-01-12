"""
Обработчик команды /start и главного меню
"""
from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from keyboards.main_keyboards import get_main_menu
from states.user_states import AnalysisStates

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    """Обработка команды /start"""
    await state.clear()  # Очистка предыдущих состояний
    
    welcome_text = (
        f"Привет, {message.from_user.first_name}!\n\n"
        "📈 Я помогу проанализировать акции и спрогнозировать их цену.\n\n"
        "Я использую для прогноза модели машинного обучения ML:\n"
        "\n\n"
        "Нажми <b>Начать анализ</b>"
    )
    
    await message.answer(
        welcome_text,
        reply_markup=get_main_menu(),
        parse_mode="HTML"
    )


@router.message(F.text == "<b>Помощь</b>")
async def cmd_help(message: Message):
    """Обработка кнопки Помощь"""
    help_text = (
        "<b>Как пользоваться:</b>\n\n"
        "1. Нажми <b>Начать анализ</b>\n"
        "2. Введи тикер (AAPL, MSFT, TSLA)\n"
        "3. Введи сумму ($)\n"
        "4. Получи прогноз\n\n"
        "Результаты — учебные."
    )
    
    await message.answer(help_text, parse_mode="HTML")


@router.message(F.text == "История")
async def cmd_history(message: Message):
    """Обработка кнопки История"""
    await message.answer("История в разработке...")
