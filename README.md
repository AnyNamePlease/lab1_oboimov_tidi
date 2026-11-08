# Лабораторная работа: Диагностика обучения MLP

## Установка окружения
1. Создайте виртуальное окружение и активируйте его:
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # или venv\Scripts\activate  # Windows

2. Установите зависимости:
   pip install -r requirements.txt

## Запуск лабораторной работы
Запустите основной скрипт:
python main.py

## Структура генерируемых файлов
- `results/metrics_history.csv` — значения train/val loss и accuracy для 12 запусков (эпохи 0–100).
- `results/gradient_norms.csv` — нормы градиентов для конфигураций A, C, D (эпохи 0, 20, 100).
- `figures/` — графики рассеяния данных, динамики потерь/точности, норм градиентов и лучшей модели.
