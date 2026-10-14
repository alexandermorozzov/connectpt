# Скрипты и их назначение

Скрипты запускаются из корня репозитория в окружении `.venv`.

| Скрипт | Назначение |
|---|---|
| `run.py` | Обучение или поиск с перебором методов, городов, весов целей и seed. |
| `generate_dataset.py` | Подготовка датасета для обучения. |
| `make_init_routes.py` | Подготовка начальной сети города. |
| `launch.ps1` / `launch.sh` | Фоновый запуск обучения или поиска на Windows / Linux. |

Конфиги находятся в `connectpt/routes_generator/cfg/`. Описание — в [cfg/README.md](../connectpt/routes_generator/cfg/README.md).

## 1. Подготовка данных

Подготовка нужна только при отсутствии готовых данных.

Обучающий и тестовый датасеты construction:

```bash
python scripts/generate_dataset.py --config-name runs/generate_construction --out datasets/construction/20_nodes/mixed
python scripts/generate_dataset.py --config-name runs/generate_construction --out datasets/construction/20_nodes/test dataset_build.n_graphs=128 run.seed=1
```

Датасет edit содержит начальные сети. Для его подготовки нужны веса construction-модели из конфига `runs/generate_edit`.

```bash
python scripts/generate_dataset.py --config-name runs/generate_edit --out datasets/lc_copytiers_n1000_n50_r12_len8_15_v1
```

Для поиска нужны файлы города в `datasets/instances/<имя>/`: `Coords.txt`, `TravelTimes.txt`, `Demand.txt` и `Routes.txt`.
Конфиг города находится в `cfg/city/instance/<имя>.yaml`.
Начальная сеть при отсутствии `Routes.txt`:

```bash
python scripts/make_init_routes.py --instances mandl --generator nikolic --cpu
```

Генератор выбирается группой `network/initial`: `john`, `nikolic` или `lc`. Флаг `--generator` меняет выбор. Для `lc` нужны веса construction-модели.

## 2. Запуск эксперимента

Обучение construction-политики:

```bash
python scripts/run.py --config-name runs/train_construction training/algorithm=ppo
```

Обучение edit-политики:

```bash
python scripts/run.py --config-name runs/train_edit
```

Поиск:

```bash
python scripts/run.py --config-name runs/search --methods ea --instances mandl --alpha 0.5 --seeds 0
```

Доступные методы: `ea`, `nea`, `nea_edit`, `nea_combined`.
Нейросетевым методам нужны веса моделей из `cfg/search/method/<метод>.yaml`.
Флаг `--dry-run` проверяет конфиг и необходимые входы без запуска эксперимента.

## 3. Параметры

В `run.py` параметры задаются после имени конфига:

```bash
python scripts/run.py --config-name runs/train_construction training/algorithm=d3po run.seed=1 train_loop.n_iterations=200
```

| Параметр | Назначение |
|---|---|
| `training/algorithm=ppo` или `d3po` | Алгоритм обучения. |
| `run.name=имя`, `run.seed=N` | Имя и seed запуска. |
| `run.cpu=true` | Обучение на CPU. |
| `dataset.path=DIR` | Обучающий датасет. |
| `dataset.test_path=DIR` | Тестовый датасет construction. |
| `train_loop.n_iterations=N` | Число итераций обучения. |
| `paths.init_checkpoint_path=FILE` | Веса для дообучения, относительно корня весов. |

Для поиска доступны флаги `--methods`, `--instances`, `--alpha`, `--seeds`, `--iters` и `--cpu`.
Несколько значений одного флага задают перебор сочетаний.
Полный список параметров каждого скрипта: `python scripts/<скрипт>.py --help`.

## 4. Фоновый запуск

Для каждой системы предусмотрен свой launcher. Первый аргумент — Python-скрипт.
Остальные аргументы передаются этому скрипту.

Windows:

```powershell
.\scripts\launch.ps1 run.py --config-name runs/train_edit
.\scripts\launch.ps1 run.py --config-name runs/search --methods ea --instances mandl
```

Linux:

```bash
bash scripts/launch.sh run.py --config-name runs/train_edit
bash scripts/launch.sh run.py --config-name runs/search --methods ea --instances mandl
```

По умолчанию launcher возвращает управление сразу. Флаг `--wait` перед именем скрипта включает ожидание завершения.
Launcher выводит PID, путь лога и команды просмотра и остановки процесса.

## 5. Папки

| Флаг | Назначение | По умолчанию |
|---|---|---|
| `--runs-dir` | Корень папок прогонов. | `artifacts/runs` |
| `--weights-dir` | Корень весов моделей. | `artifacts/model_weights` |
| `--log-dir` / `--log` | Папка логов / отдельный файл лога. | `artifacts/logs` |
| `--out-dir` | Результаты поиска. | `artifacts/results/search` |
| `--out` | Датасет `generate_dataset.py`. | Обязательный параметр. |

Графики доступны в общем TensorBoard с `--logdir artifacts`.
