"""
Обработчик ввода суммы инвестиции
"""
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

from states.user_states import AnalysisStates
from keyboards.main_keyboards import get_confirmation_keyboard, get_main_menu, get_cancel_keyboard
from utils.validators import validate_amount

router = Router()


@router.message(AnalysisStates.waiting_for_amount, F.text == "❌ Отменить")
async def cancel_amount_input(message: Message, state: FSMContext):
    """Отмена ввода суммы"""
    await state.clear()
    await message.answer(
        "❌ Анализ отменён.",
        reply_markup=get_main_menu()
    )


@router.message(AnalysisStates.waiting_for_amount)
async def process_amount(message: Message, state: FSMContext):
    """Обработка введённой суммы"""
    amount_str = message.text.strip()
    
    # Валидация суммы
    is_valid, error_message, amount = validate_amount(amount_str)
    
    if not is_valid:
        await message.answer(
            f"❌ {error_message}\n\n"
            "Попробуйте снова или нажмите <b>❌ Отменить</b>.",
            reply_markup=get_cancel_keyboard(),  # ← ФИКС: Добавлена клавиатура
            parse_mode="HTML"
        )
        return
    
    # Сохранение суммы в состояние
    await state.update_data(amount=amount)
    await state.set_state(AnalysisStates.confirmation)
    
    # Получение данных пользователя
    user_data = await state.get_data()
    ticker = user_data['ticker']
    
    # Форматирование суммы с разделителями
    formatted_amount = f"{amount:,}".replace(',', ' ')
    
    confirmation_text = (
        "✅ Данные приняты!\n\n"
        "<b>Проверьте введённые данные:</b>\n\n"
        f"Тикер: <b>{ticker}</b>\n"
        f"Сумма: <b>${formatted_amount}</b>\n\n"
        "Всё верно?"
    )
    
    await message.answer(
        confirmation_text,
        reply_markup=get_confirmation_keyboard(),
        parse_mode="HTML"
    )


@router.callback_query(F.data == "confirm_analysis", AnalysisStates.confirmation)
async def confirm_analysis(callback: CallbackQuery, state: FSMContext):
    """Подтверждение и запуск анализа"""
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)  # Убираем кнопки
    
    # Переход к обработке
    await state.set_state(AnalysisStates.processing)
    
    # Получение данных
    user_data = await state.get_data()
    ticker = user_data['ticker']
    amount = user_data['amount']
    
    # Сообщение о начале обработки
    await callback.message.answer(
        f"<b>Загружаю исторические данные для {ticker}...</b>\n\n"
        "Это может занять несколько секунд.",
        parse_mode="HTML"
    )
    
    # Импорт и запуск анализа
    from handlers.analysis_handler import run_analysis
    await run_analysis(callback.message, state, ticker, amount)


@router.callback_query(F.data == "cancel_analysis", AnalysisStates.confirmation)
async def cancel_analysis(callback: CallbackQuery, state: FSMContext):
    """Отмена анализа"""
    await callback.answer("Анализ отменён")
    await callback.message.edit_reply_markup(reply_markup=None)
    
    await state.clear()
    
    await callback.message.answer(
        "❌ Анализ отменён. Можете начать заново.",
        reply_markup=get_main_menu()
    )

