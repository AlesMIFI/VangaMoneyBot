"""
Клавиатуры бота
"""
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton


def get_main_menu():
    """Главное меню"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Начать анализ")],
            [KeyboardButton(text="Помощь"), KeyboardButton(text="История")]
        ],
        resize_keyboard=True
    )
    return keyboard


def get_confirmation_keyboard():
    """Клавиатура подтверждения"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвердить", callback_data="confirm_analysis"),
                InlineKeyboardButton(text="❌ Отменить", callback_data="cancel_analysis")
            ]
        ]
    )
    return keyboard


def get_cancel_keyboard():
    """Кнопка отмены"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="❌ Отменить")]
        ],
        resize_keyboard=True
    )
    return keyboard
