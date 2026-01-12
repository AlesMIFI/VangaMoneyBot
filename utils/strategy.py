"""
Торговые рекомендации и расчет прибыли
"""
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


def find_trading_points(forecast: np.ndarray, min_distance: int = 3) -> Tuple[List[int], List[int]]:
    """
    Поиск локальных минимумов (покупка) и максимумов (продажа)
    
    Аргументы:
        forecast: Массив прогнозных цен
        min_distance: Минимальное расстояние между пиками
        
    Возвращает:
        (buy_indices, sell_indices)
    """
    logger.info("Finding trading points...")
    
    # Метод 1: scipy.signal.find_peaks с адаптивными параметрами
    # Рассчитываем стандартное отклонение для определения prominence
    std_dev = np.std(forecast)
    prominence = std_dev * 0.3  # 30% от стандартного отклонения
    
    # Поиск локальных максимумов (продавать)
    peaks, _ = find_peaks(forecast, prominence=prominence, distance=min_distance)
    
    # Поиск локальных минимумов (покупать)
    troughs, _ = find_peaks(-forecast, prominence=prominence, distance=min_distance)
    
    # Если не нашли точек, используем альтернативный метод
    if len(peaks) == 0 and len(troughs) == 0:
        logger.info("No peaks found with find_peaks, trying argrelextrema...")
        
        # Альтернативный метод: argrelextrema (менее строгий)
        peaks = argrelextrema(forecast, np.greater, order=2)[0]
        troughs = argrelextrema(forecast, np.less, order=2)[0]
    
    # Если все еще нет точек, создаем базовые рекомендации
    if len(peaks) == 0 and len(troughs) == 0:
        logger.warning("No trading points found, creating basic recommendations")
        
        # Простая стратегия: покупка в начале, продажа в конце
        # Находим самую низкую и самую высокую точки
        min_idx = np.argmin(forecast)
        max_idx = np.argmax(forecast)
        
        if min_idx < max_idx:
            troughs = [min_idx]
            peaks = [max_idx]
        else:
            # Если максимум раньше минимума, просто берем начало и конец
            troughs = [0]
            peaks = [len(forecast) - 1]
    
    logger.info(f"Found {len(troughs)} buy points and {len(peaks)} sell points")
    
    return troughs.tolist(), peaks.tolist()


def calculate_profit(
    forecast: np.ndarray,
    buy_points: List[int],
    sell_points: List[int],
    initial_amount: float
) -> Dict:
    """
    Расчет потенциальной прибыли по торговой стратегии
    
    Аргументы:
        forecast: Массив прогнозных цен
        buy_points: Индексы покупок
        sell_points: Индексы продаж
        initial_amount: Начальная сумма инвестиции
        
    Возвращает:
        Словарь с результатами стратегии
    """
    logger.info(f"Calculating profit with initial amount ${initial_amount}")
    
    if not buy_points or not sell_points:
        logger.warning("No trading points, using hold strategy")
        
        # Простая стратегия: купить в начале, продать в конце
        shares = initial_amount / forecast[0]
        final_value = shares * forecast[-1]
        profit = final_value - initial_amount
        return_pct = (profit / initial_amount) * 100
        
        return {
            'strategy': 'hold',
            'initial_amount': initial_amount,
            'final_value': final_value,
            'profit': profit,
            'return_pct': return_pct,
            'trades': [
                {'day': 0, 'action': 'buy', 'price': forecast[0], 'shares': shares},
                {'day': len(forecast)-1, 'action': 'sell', 'price': forecast[-1], 'shares': shares}
            ],
            'total_trades': 2
        }
    
    # Стратегия с точками покупки/продажи
    balance = initial_amount
    shares = 0
    trades = []
    
    # Объединяем и сортируем все торговые точки
    all_points = []
    for idx in buy_points:
        all_points.append((idx, 'buy'))
    for idx in sell_points:
        all_points.append((idx, 'sell'))
    
    all_points.sort(key=lambda x: x[0])
    
    # Выполняем торговые операции
    for day, action in all_points:
        price = forecast[day]
        
        if action == 'buy' and balance > 0:
            # Покупаем на всю сумму
            shares_to_buy = balance / price
            shares += shares_to_buy
            balance = 0
            trades.append({
                'day': day,
                'action': 'buy',
                'price': price,
                'shares': shares_to_buy
            })
            logger.debug(f"Day {day}: BUY {shares_to_buy:.2f} shares at ${price:.2f}")
            
        elif action == 'sell' and shares > 0:
            # Продаем все акции
            balance = shares * price
            trades.append({
                'day': day,
                'action': 'sell',
                'price': price,
                'shares': shares
            })
            logger.debug(f"Day {day}: SELL {shares:.2f} shares at ${price:.2f}")
            shares = 0
    
    # Если остались акции, продаем в конце
    if shares > 0:
        balance = shares * forecast[-1]
        trades.append({
            'day': len(forecast)-1,
            'action': 'sell',
            'price': forecast[-1],
            'shares': shares
        })
        shares = 0
    
    # Если остались деньги без акций, считаем по последней цене
    if balance == 0 and len(trades) > 0:
        last_trade = trades[-1]
        if last_trade['action'] == 'buy':
            balance = last_trade['shares'] * forecast[-1]
    
    final_value = balance
    profit = final_value - initial_amount
    return_pct = (profit / initial_amount) * 100
    
    logger.info(f"Strategy completed: profit=${profit:.2f} ({return_pct:.2f}%)")
    
    return {
        'strategy': 'peaks_troughs',
        'initial_amount': initial_amount,
        'final_value': final_value,
        'profit': profit,
        'return_pct': return_pct,
        'trades': trades,
        'total_trades': len(trades)
    }


def format_trading_recommendations(
    forecast: np.ndarray,
    buy_points: List[int],
    sell_points: List[int],
    profit_info: Dict
) -> str:
    """
    Форматирование торговых рекомендаций для отправки пользователю

    Аргументы:
        forecast: Массив прогнозов
        buy_points: Индексы покупок
        sell_points: Индексы продаж
        profit_info: Информация о прибыли

    Возвращает:
        Отформатированная строка с рекомендациями
    """
    text = "💡 <b>Торговые рекомендации:</b>\n\n"
    
    # Рекомендации по покупке
    if buy_points:
        text += "<b>Купить в дни:</b>\n"
        for idx in buy_points[:3]:  # Показываем первые 3
            text += f"   • День {idx + 1}: ${forecast[idx]:.2f}\n"
        if len(buy_points) > 3:
            text += f"   ... и еще {len(buy_points) - 3}\n"
        text += "\n"
    
    # Рекомендации по продаже
    if sell_points:
        text += "<b>Продать в дни:</b>\n"
        for idx in sell_points[:3]:  # Показываем первые 3
            text += f"   • День {idx + 1}: ${forecast[idx]:.2f}\n"
        if len(sell_points) > 3:
            text += f"   ... и еще {len(sell_points) - 3}\n"
        text += "\n"
    
    # Информация о прибыли
    text += "<b>Потенциальная прибыль:</b>\n"
    text += f"   • Начальная сумма: ${profit_info['initial_amount']:,.2f}\n"
    text += f"   • Конечная стоимость: ${profit_info['final_value']:,.2f}\n"
    
    profit_emoji = "📈" if profit_info['profit'] > 0 else "📉"
    text += f"   • Прибыль: {profit_emoji} ${profit_info['profit']:,.2f} "
    text += f"({profit_info['return_pct']:+.2f}%)\n"
    
    if profit_info.get('total_trades'):
        text += f"   • Всего сделок: {profit_info['total_trades']}\n"
    
    # Предупреждение
    text += "\n <i>Результаты носят учебный характер!</i>"
    
    return text

