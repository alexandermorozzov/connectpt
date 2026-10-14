# Конфиги

Конфиги находятся в `connectpt/routes_generator/cfg/` и собираются Hydra.
Файлы `runs/` выбирают группы параметров для конкретной задачи.
Команды запуска описаны в [scripts/README.md](../../../scripts/README.md).

## Основные конфиги

| Конфиг в `runs/` | Назначение |
|---|---|
| `search` | Поиск маршрутной сети. |
| `train_construction` | Обучение построению сети с нуля. |
| `train_edit` | Обучение редактированию существующей сети. |
| `generate_construction` | Подготовка датасета городов без начальных сетей. |
| `generate_edit` | Подготовка датасета городов с начальными сетями. |
| `init_routes` | Подготовка начальной сети; генератор выбирается через `network/initial`. |

## Группы параметров

| Папка | Назначение |
|---|---|
| `city/instance/` | Город и ограничения маршрутов. |
| `objectives/` | Цели оптимизации, штрафы и веса целей. |
| `policy/model/` | Архитектура модели. |
| `policy/model_profile/` | Модель, веса и среда для применения обученной политики. |
| `env/` | Среда construction или edit и доступные действия. |
| `training/` | Алгоритм, датасет, подготовка данных и расписание обучения. |
| `search/` | Метод поиска, начальная сеть и параметры поиска. |
| `network/initial/`, `network/corruption/` | Генератор начальной сети и параметры её порчи. |
| `common/experiment.yaml` | Общие настройки маршрутов и seed эксперимента. |

## Выбор параметров

Группа выбирается по пути к YAML: `training/algorithm=d3po`.
Отдельное значение меняется по имени поля собранного конфига: `algorithm.gamma=0.95`.
Параметры запуска передаются в виде `ключ=значение`:

```bash
python scripts/run.py --config-name runs/train_construction training/algorithm=d3po run.seed=1
python scripts/run.py --config-name runs/train_edit dataset.path=datasets/edit run.cpu=true
```

Веса моделей задаются в `checkpoint_path` относительно корня `--weights-dir`.
Папки результатов и логов задаются флагами скриптов запуска.

## Проверка конфига

Флаг `--dry-run` проверяет конфиг и необходимые входы без запуска эксперимента:

```bash
python scripts/run.py --config-name runs/train_edit --dry-run
python scripts/run.py --config-name runs/search --methods ea --instances mandl --dry-run
```
