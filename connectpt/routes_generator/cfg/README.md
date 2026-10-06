# Устройство конфигов

Конфиг собирается Hydra из небольших пресетов. Точки входа — файлы в `runs/`.
Они выбирают пресеты и задают только настройки прогона.

```text
cfg/
├── runs/            # точки входа: search, train, train_construction, train_construction_overnight, prepare_dataset
├── instance/        # путь к бенчмарку и ограничения маршрутов
├── method/          # методы поиска: ea, nea, nea_edit, nea_combined
├── bee_set/         # набор пчёл метода и их число
├── bee/             # один тип пчелы (оператор)
├── cost/            # цель: rtt_wmc (поиск, edit), att_rtt (construction)
├── adjustment/      # штраф за изменение сети: cap, target, disabled
├── model/           # архитектура сети
├── model_profile/   # версия модели: архитектура + чекпоинт
├── dataset/         # датасет и его разбиение
├── dataset_build/   # параметры генерации датасета
├── curriculum/      # расписание тиров edit-обучения
├── trainer/         # оптимизатор и цикл PPO / D3PO
├── experiment/      # общие настройки: seed, symmetric_routes
├── sweep/           # сетка поиска: α, число итераций
└── stacks/          # внутренняя сборка для оценки моделей
```

- `model/` отвечает, **какая архитектура** строится.
- `model_profile/` отвечает, **какая версия** архитектуры используется, вместе
  с чекпоинтом.
- Метод выбирает профили по роли: `construction` или `edit`.

Примеры:

```powershell
python scripts/run_search.py --dry-run --methods ea
python scripts/run_search.py --methods nea_edit --instances mandl
python scripts/run_training.py --dry-run
python scripts/run_training.py --config-name runs/train_construction trainer=d3po_construction --dry-run
python scripts/prepare_data.py
```

Поиск и edit-обучение используют одну цель `cost=rtt_wmc`. Штраф за изменение
сети задаётся отдельно: `adjustment=cap`, `adjustment=target` или
`adjustment=disabled`. Прогоны по умолчанию берут `cap`.

YAML содержит только значения. Комментарии в конфигах не пишутся.
