
"""
Главный файл Telegram бота для прогнозирования акций
"""
import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from config import BOT_TOKEN
from handlers import start_handler, ticker_handler, amount_handler, analysis_handler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Запуск бота"""
    # Инициализация бота и диспетчера
    bot = Bot(token=BOT_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    
    # Регистрация обработчиков
    dp.include_router(start_handler.router)
    dp.include_router(ticker_handler.router)
    dp.include_router(amount_handler.router)
    dp.include_router(analysis_handler.router)
    
    logger.info("Бот запущен!")
    
    # Запуск polling
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


if __name__ == '__main__':
    # Фикс для Windows - использовать SelectorEventLoop
    if sys.platform == 'win32':
        # Для Windows используем SelectorEventLoop вместо ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
