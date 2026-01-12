# VangaMoneyBot

**Telegram-бот для прогнозирования цен акций с использованием машинного обучения и нейронных сетей**

## Описание проекта

**VangaMoneyBot** — это Telegram-бот для анализа и прогнозирования цен акций на основе временных рядов. Бот обучает 5 различных моделей (ML, статистические, нейронные сети), выбирает лучшую и предоставляет:

- Прогноз цены на 30 дней вперед
- График с историей и прогнозом
- Торговые рекомендации (точки покупки/продажи)
- Расчет потенциальной прибыли
- Сравнение 5 моделей машинного обучения


## Используемые модели

1. **Random Forest** (ML) - Случайный лес
2. **ARIMA** (Статистика) - Авторегрессионная модель
3. **Prophet** (Meta) - Прогнозирование временных рядов
4. **LSTM** (PyTorch) - Рекуррентная нейросеть
5. **GRU** (PyTorch) - Упрощенная версия LSTM

## Структура проекта
VangaMoneyBot/
│
├── bot.py # Главный файл бота (запуск)
├── config.py # Конфигурация и настройки
├── requirements.txt # Зависимости проекта
├── README.md # Описание проекта
├── .env # Переменные окружения (токен)
├── .gitignore # Исключения для Git
│
├── handlers/ # Обработчики команд и сообщений
│ ├── init.py
│ ├── start_handler.py # /start, приветствие, кнопки
│ ├── ticker_handler.py # Обработка ввода тикера
│ ├── amount_handler.py # Обработка ввода суммы
│ └── analysis_handler.py # Запуск анализа
│
├── models/ # ML/Stat/DL модели
│ ├── init.py
│ ├── base_model.py # Базовый класс для всех моделей
│ ├── random_forest.py # Random Forest модель
│ ├── arima_model.py # ARIMA модель
│ ├── prophet_model.py # Prophet модель
│ ├── lstm_model.py # LSTM модель (PyTorch)
│ └── gru_model.py # GRU модель (PyTorch)
│
├── utils/ # Вспомогательные функции
│ ├── data_loader.py # Загрузка данных с yfinance
│ ├── data_preprocessor.py # Предобработка данных
│ ├── validators.py # Валидация тикера и суммы
│ ├── model_selector.py # Обучение и выбор лучшей модели
│ ├── visualizer.py # Генерация графиков
│ ├── strategy.py # Торговые рекомендации, расчет прибыли
│ └── logger.py # Логирование запросов
│
├── keyboards/ # Клавиатуры для бота
│ └── main_keyboards.py # Кнопки интерфейса
│
├── states/ # FSM состояния
│ └── user_states.py # Состояния диалога с пользователем
│
├── data/ # Данные (создается автоматически)
│ └── logs.csv # Журнал логов
│
└── temp/ # Временные файлы (графики)
└── plots/ # Сохраненные графики

# Требования

- Python 3.9+
- Telegram аккаунт
- Интернет соединение (для загрузки данных с Yahoo Finance)


## Установка и запуск

### **Клонирование репозитория**

```bash
git clone https://github.com/yourusername/VangaMoneyBot.git
cd VangaMoneyBot


### 2. Установка зависимостей
Создайте виртуальное окружение (рекомендуется):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate      # Windows
Установите зависимости:

```bash
pip install -r requirements.txt

#### Получение токена бота
Шаг 1: Найдите @BotFather в Telegram
Откройте Telegram

В поиске введите: @BotFather
Убедитесь, что это официальный бот с галочкой 

Шаг 2: Создайте бота
Отправьте команду /newbot или выберете в меню
BotFather спросит имя бота (например: VangaMoneyBot)
Затем username (должен заканчиваться на bot, например: VangaMoney_bot) 
Вы получите токен вида: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

### Настройка токена
Создайте файл .env в корне проекта:
В файле сохраните строку:
BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz(ваш токен)

Запуск бота
```bash
python bot.py


### Технологии
### Backend
* aiogram 3.13 - Асинхронный фреймворк для Telegram Bot API
* yfinance 0.2.51 - Загрузка исторических данных акций

###Machine Learning
- scikit-learn 1.5.2 - Random Forest
- statsmodels 0.14.4 - ARIMA
- prophet 1.1.6 - Prophet (Facebook)
- torch 2.6+ - LSTM/GRU (PyTorch)

### Data & Visualization
- pandas 2.2.3 - Обработка временных рядов
- numpy 2.1.3 - Математические операции
- matplotlib 3.9.2 - Графики прогнозов
- scipy 1.14.1 - Поиск локальных экстремумов

###Как это работает?
Пользователь вводит тикер (например, AAPL)
Загрузка данных с Yahoo Finance (2 года истории)
Обучение 5 моделей параллельно:Random Forest, ARIMA, Prophet, LSTM, GRU

Выбор лучшей по метрике RMSE
Прогноз на 30 дней
Поиск точек покупки/продажи (локальные min/max)
Расчет потенциальной прибыли
Генерация графика
Логирование в data/logs.csv

#Важно
Учебный проект - не используйте для реальной торговли!
Токен бота - храните в .env, не публикуйте в Git
### Данные - Yahoo Finance может блокировать при частых запросах
Prophet - может не установиться на Windows, закомментируйте импорты

Решение проблем
Проблема: curl_cffi не установлен

bash
pip install curl_cffi

Проблема: Prophet не устанавливается
Закомментируйте в models/__init__.py:

# from .prophet_model import ProphetModel

Проблема: yfinance не загружает данные
Проверьте интернет-соединение или используйте VPN

