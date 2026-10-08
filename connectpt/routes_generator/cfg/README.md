# Устройство конфигов

Конфиг собирается Hydra из небольших пресетов. Точки входа — файлы в `runs/`.
Они выбирают пресеты и задают только настройки прогона.

```text
cfg/
├── runs/            # точки входа: search, train, train_construction, train_construction_overnight, prepare_dataset, compare
├── instance/        # путь к бенчмарку и ограничения маршрутов
├── method/          # методы поиска: ea, nea, nea_edit, nea_combined
├── bee_set/         # набор пчёл метода и их число
├── bee/             # один тип пчелы (оператор)
├── cost/            # цель: rtt_wmc (поиск, edit), att_rtt (construction)
├── adjustment/      # штраф за изменение сети: cap
├── model/           # архитектура сети
├── model_profile/   # версия модели: архитектура + чекпоинт
├── dataset/         # датасет и его разбиение
├── dataset_build/   # параметры генерации датасета
├── curriculum/      # расписание тиров edit-обучения
├── algorithm/       # PPO / D3PO и параметры критика
├── features/        # порядок и размерности признаков
├── experiment/      # общие настройки: seed, symmetric_routes
├── sweep/           # точка поиска: α, число итераций
└── env/             # construction / edit: награда и ограничения действий
```

- `model/` отвечает, **какая архитектура** строится.
- `model_profile/` отвечает, **какая версия** архитектуры используется, вместе
  с чекпоинтом.
- Метод выбирает профили по роли: `construction` или `edit`.
- `algorithm` задаёт GAE, clip, entropy, diversity и критик. `train_loop` задаёт
  batch, horizon, число итераций, валидацию и бюджет времени. `optimizer` задаёт
  параметры оптимизатора. Обе среды используют один цикл обучения.

Примеры:

```powershell
python scripts/run_search.py --dry-run --methods ea
python scripts/run_search.py --methods nea_edit --instances mandl
python scripts/run.py --config-name runs/train --dry-run
python scripts/run.py --config-name runs/train_construction algorithm=d3po --dry-run
python scripts/run.py --config-name runs/prepare_dataset
```

Поиск и edit-обучение используют одну цель `cost=rtt_wmc`. Штраф за изменение
сети задаёт `adjustment/cap.yaml`. Выключить штраф: `cost.kwargs.penalties.adjustment=0`
(в поиске — `--adj-weight 0`). Режим «цель»: `cost.kwargs.adjustment_degree_objective=target`.

YAML содержит только значения. Комментарии в конфигах не пишутся.
