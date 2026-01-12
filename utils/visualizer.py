"""
Генерация графиков прогнозов
"""
import matplotlib
matplotlib.use('Agg')  # Использовать non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import TEMP_PLOTS_DIR
import logging

logger = logging.getLogger(__name__)

# Настройка matplotlib
plt.rcParams['figure.figsize'] = (10, 5)  # уменьшили размер
plt.rcParams['font.size'] = 9


def generate_forecast_plot(
    df: pd.DataFrame,
    forecast: np.ndarray,
    ticker: str,
    best_model_name: str,
    buy_points: list = None,
    sell_points: list = None
) -> str:
    """
    Генерация графика с историей и прогнозом
    
    Аргументы:
        df: DataFrame с историческими данными [date, price]
        forecast: Массив прогнозных значений
        ticker: Тикер акции
        best_model_name: Название лучшей модели
        buy_points: Индексы точек покупки в прогнозе
        sell_points: Индексы точек продажи в прогнозе

    Возвращает:
        Путь к сохраненному файлу
    """
    logger.info(f"Generating plot for {ticker}")
    
    # Создание фигуры (меньший размер)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    
    # Исторические данные (берем последние 90 дней для компактности)
    historical_dates = df['date'].values[-90:]
    historical_prices = df['price'].values[-90:]
    
    # Даты для прогноза
    last_date = df['date'].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(forecast),
        freq='D'
    )
    
    # График исторических данных
    ax.plot(
        historical_dates,
        historical_prices,
        label='История',
        color='#2E86AB',
        linewidth=1.5
    )
    
    # График прогноза
    ax.plot(
        forecast_dates,
        forecast,
        label=f'Прогноз ({best_model_name})',
        color='#A23B72',
        linewidth=1.5,
        linestyle='--'
    )
    
    # Точка перехода
    ax.scatter(
        [last_date],
        [historical_prices[-1]],
        color='green',
        s=50,
        zorder=5
    )
    
    # Точки покупки
    if buy_points:
        buy_dates = [forecast_dates[i] for i in buy_points]
        buy_prices = [forecast[i] for i in buy_points]
        ax.scatter(
            buy_dates,
            buy_prices,
            color='green',
            marker='^',
            s=100,
            zorder=5,
            label='Купить'
        )
    
    # Точки продажи
    if sell_points:
        sell_dates = [forecast_dates[i] for i in sell_points]
        sell_prices = [forecast[i] for i in sell_points]
        ax.scatter(
            sell_dates,
            sell_prices,
            color='red',
            marker='v',
            s=100,
            zorder=5,
            label='Продать'
        )
    
    # Форматирование
    ax.set_xlabel('Дата', fontsize=10)
    ax.set_ylabel('Цена ($)', fontsize=10)
    ax.set_title(
        f'{ticker} | {best_model_name}',
        fontsize=11,
        fontweight='bold'
    )
    
    # Форматирование оси X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=8)
    
    # Tight layout
    plt.tight_layout()
    
    # Сохранение (меньший DPI и качество)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{ticker}_{timestamp}.png'
    filepath = os.path.join(TEMP_PLOTS_DIR, filename)
    
    plt.savefig(filepath, dpi=72, bbox_inches='tight', format='png', facecolor='white',
    edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    
    # Дополнительно: сжать файл после сохранения
    try:
        from PIL import Image
        img = Image.open(filepath)
        img = img.convert('RGB')
        img.save(filepath, 'PNG', optimize=True, quality=85)
        logger.info(f"Image compressed: {filepath}")
    except:
        pass
    logger.info(f"Plot saved to {filepath}")
    
    return filepath

