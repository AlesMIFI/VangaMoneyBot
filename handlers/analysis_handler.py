"""
Обработчик процесса анализа акций
"""
from aiogram import Router
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
import logging
import asyncio
import os

from utils.data_loader import load_stock_data, get_stock_info
from utils.model_selector import ModelSelector
from utils.visualizer import generate_forecast_plot
from utils.strategy import find_trading_points, calculate_profit, format_trading_recommendations
from utils.logger import log_analysis
from keyboards.main_keyboards import get_main_menu

router = Router()
logger = logging.getLogger(__name__)


async def run_analysis(message: Message, state: FSMContext, ticker: str, amount: int):
    """
    Запуск полного анализа акций
    
    Аргументы:
        message: Сообщение пользователя
        state: Состояние FSM
        ticker: Тикер акции
        amount: Сумма инвестиции
    """
    user_id = message.from_user.id
    
    try:
        # 1. Загрузка данных
        df, error_msg = await load_stock_data(ticker) # загрузка данных с обработкой кэша
        
        if df is None:
            await message.answer(
                f"❌ {error_msg}",
                reply_markup=get_main_menu(),
                parse_mode="HTML"
            )
            await state.clear()
            return
        
        # 2. Информация о компании
        stock_info = await get_stock_info(ticker)
        
        await message.answer(
            f"✅ Данные загружены!\n\n"
            f"<b>{stock_info['name']}</b>\n"
            f"Сектор: {stock_info['sector']}\n"
            f"Записей: {len(df)} дней\n\n"
            f"Начинаю обучение моделей, пожалуйста подождите...",
            parse_mode="HTML"
        )
        
        # 3. Обучение моделей в отдельном потоке
        selector = ModelSelector(df)
        selector.prepare_data()
        
        results = await asyncio.to_thread(selector.train_all_models)
        
        # 4. Выбор лучшей модели
        best_name, best_result, best_metrics = selector.select_best_model()
        
        await message.answer(
            f"✅ Обучение завершено!\n\n"
            f"<b>Лучшая модель: {best_name}</b>\n\n"
            f"<b>Метрики:</b>\n"
            f"   • RMSE: {best_metrics['rmse']:.4f}\n"
            f"   • MAPE: {best_metrics['mape']:.2f}%\n\n"
            f"Строю прогноз, ожидайте..",
            parse_mode="HTML"
        )
        
        # 5. Финальный прогноз на 30 дней
        forecast = await asyncio.to_thread(selector.retrain_and_forecast)
        
        # 6. Поиск торговых точек
        buy_points, sell_points = find_trading_points(forecast)
        
        # 7. Расчет прибыли
        profit_info = calculate_profit(forecast, buy_points, sell_points, amount)
        
        # 8. Генерация графика
        plot_path = await asyncio.to_thread(
            generate_forecast_plot,
            df, forecast, ticker, best_name, buy_points, sell_points
        )
        
        # 9. Основные результаты
        current_price = df['price'].iloc[-1]
        forecast_price = forecast[-1]
        change_percent = ((forecast_price - current_price) / current_price) * 100
        
        results_text = (
            f"✅<b>Прогноз готов!</b>\n\n"
            f"<b>Результаты:</b>\n"
            f"   • Текущая цена: ${current_price:.2f}\n"
            f"   • Прогноз (30 дней): ${forecast_price:.2f}\n"
            f"   • Изменение: {change_percent:+.2f}%\n"
        )
        
        await message.answer(results_text, parse_mode="HTML")
        
        # 10. Отправка графика с retry и увеличенным timeout
        max_retries = 5  # Увеличили с 3 до 5
        retry_delay = 3   # Секунды между попытками

        for attempt in range(max_retries):
            try:
                photo = FSInputFile(plot_path)
        
                # Отправляем с увеличенным таймаутом
                await message.answer_photo(
                    photo,
                    caption=f" {ticker} | {best_name}",
                    request_timeout=30  #  timeout 30 секунд
                )
        
                logger.info(f"✅ Photo sent successfully on attempt {attempt + 1}")
                break  # Успешно отправлено
        
            except Exception as e:
                logger.warning(f"Failed to send photo (attempt {attempt+1}/{max_retries}): {e}")
        
                if attempt == max_retries - 1:
                    # Последняя попытка не удалась - отправляем текст
                    await message.answer(
                    "Не удалось отправить график (проблема сети).\n"
                    "Показываю текстовые результаты.",
                    parse_mode="HTML"
                )
            else:
                # Ждем перед следующей попыткой
                await asyncio.sleep(retry_delay)

        
        # 11. Торговые рекомендации
        recommendations_text = format_trading_recommendations(
            forecast, buy_points, sell_points, profit_info
        )
        
        await message.answer(recommendations_text, parse_mode="HTML")
        
        # 12. Таблица сравнения моделей
        comparison_df = selector.get_comparison_table()
        table_text = "<b>Сравнение моделей:</b>\n\n<pre>" + comparison_df.to_string(index=False) + "</pre>"
        
        await message.answer(table_text, parse_mode="HTML")
        
        # 13. Логирование
        log_analysis(
            user_id=user_id,
            ticker=ticker,
            amount=amount,
            best_model=best_name,
            rmse=best_metrics['rmse'],
            mape=best_metrics['mape'],
            predicted_profit=profit_info['profit'],
            recommendation='buy' if profit_info['profit'] > 0 else 'hold'
        )
        
        # 14. Очистка временного файла
        try:
            os.remove(plot_path)
        except:
            pass
        
        await message.answer(
            "✅ Анализ завершен!",
            reply_markup=get_main_menu()
        )
        
        await state.clear()
        
    except Exception as e:
        logger.error(f"Ошибка в run_analysis для пользователя {user_id}: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при анализе.\n\n"
            f"Попробуйте позже.",
            reply_markup=get_main_menu()
        )
        await state.clear()
