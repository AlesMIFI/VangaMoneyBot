"""
Конфигурация проекта
"""
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# Токен бота
BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')

# Настройки данных
DATA_PERIOD = '2y'  # Период загрузки данных (2 года)
FORECAST_DAYS = 30  # Дни прогноза
TRAIN_TEST_SPLIT = 0.75  # 75% train, 25% test

# Настройки моделей
MODELS_TO_USE = [
    'RandomForest',
    'ARIMA',
    'Prophet',
    'LSTM',
    'GRU'
]

# Метрики для сравнения
METRICS = ['RMSE', 'MAPE', 'MAE']

# Настройки логирования
LOG_FILE = 'data/logs.csv'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Настройки кэширования, чтобы избежать повторных вычислений
#CACHE_ENABLED = False
#CACHE_TTL = 3600  # Время жизни кэша (1 час)

# Лимиты валидации для тикеров и сумм
TICKER_MAX_LENGTH = 5
TICKER_MIN_LENGTH = 1
AMOUNT_MIN = 1
AMOUNT_MAX = 1_000_000_000

# Пути к временным файлам
TEMP_PLOTS_DIR = 'temp/plots/'
#CACHE_DIR = 'data/cache/'

# Создание необходимых директорий после определения всех переменных
os.makedirs(TEMP_PLOTS_DIR, exist_ok=True)
#os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)
