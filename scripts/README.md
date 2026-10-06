# Запуск

Все команды выполняйте из корня репозитория.

| Скрипт | Назначение |
|---|---|
| `run_search.py` | Оптимизация сети: EA, NEA, NEA-Edit, NEA-Combined. |
| `make_init_routes.py` | Строит начальные сети бенчмарков. Запускается один раз. |
| `data_plot.py` | Рисует маршруты поверх УДС. |
| `prepare_data.py` | Генерирует датасет для edit-обучения. |
| `run_training.py` | Обучает edit- или construction-политику. |
| `run_search.ps1` / `.sh`, `run_training.ps1` / `.sh` | То же в фоне, лог в `artifacts/logs/`. |
| `run_construction_comparison.ps1` | Очередь construction PPO / D3PO по одному. |
| `run_construction_overnight.ps1` | PPO и D3PO параллельно на ночь. |

Конфиги лежат в `connectpt/routes_generator/cfg/`. Точки входа — `cfg/runs/*.yaml`.
Устройство конфигов — `cfg/README.md`.

## 1. Поиск

```
python scripts/run_search.py [--methods ...] [--instances ...] [--alpha ...] [--seeds ...]
                             [--iters N] [--adj-weight W] [--adj-target T]
                             [--out-dir DIR] [--runs-dir DIR] [--weights-dir DIR]
                             [--log-dir DIR] [--log FILE] [--cpu] [--no-initial] [--dry-run]
```

Без аргументов скрипт делает один короткий прогон.

| Флаг | По умолчанию |
|---|---|
| `--methods` | `nea_combined` |
| `--instances` | `mandl` |
| `--alpha` | `0.5` |
| `--seeds` | `0` |
| `--iters` | из `cfg/sweep/single.yaml` (400) |
| `--adj-weight`, `--adj-target` | из `cfg/adjustment/cap.yaml` (вес 10, цель 0.2) |
| `--out-dir` | `artifacts/results/search` |

`--adj-weight 0` выключает штраф за изменение сети. `--adj-target` — доля, а не проценты.

Валидация статьи — сетку задайте явно:

```bash
python scripts/run_search.py --methods ea nea nea_edit nea_combined --instances mandl mumford0 mumford1 mumford2 mumford3 --alpha 0 0.25 0.5 0.75 1 --seeds 0 1 2 3 4 --adj-weight 0 --out-dir artifacts/results/validation
```

Быстрая проверка без вычислений:

```bash
python scripts/run_search.py --dry-run --methods ea nea nea_edit nea_combined
```

Результат для каждого инстанса:

- `search_<инстанс>.csv` — строка на точку сетки и строки `Initial`;
- `search_<инстанс>_<метод>_seed<N>_routes.pt` — маршруты.

Префикс `search` — имя конфига (`--config`).

### Методы

| `--methods` | Оператор 1 | Оператор 2 | Модели |
|---|---|---|---|
| `ea` | замена маршрута кратчайшим путём | правка концов | — |
| `nea` | нейросетевая перестройка | правка концов | construction |
| `nea_edit` | замена кратчайшим путём | обученный trim / extend | edit |
| `nea_combined` | нейросетевая перестройка | обученный trim / extend | construction + edit |

Состав пчёл и модели метода — `cfg/method/<метод>.yaml`. Цель — `cfg/cost/rtt_wmc.yaml`.

## 2. Инстансы

Один инстанс — одна папка `datasets/instances/<имя>/`:

| Файл | Содержимое |
|---|---|
| `Coords.txt` | первая строка — число узлов N, затем N строк `x y` |
| `TravelTimes.txt` | матрица N×N **в минутах** (код переводит в секунды) |
| `Demand.txt` | матрица корреспонденций N×N |
| `Routes.txt` | начальная сеть: строка — маршрут, узлы через пробел |

Рядом с папкой нужен файл `cfg/instance/<имя>.yaml`:

```yaml
n_routes: 6
min_route_len: 2
max_route_len: 8
```

Имя папки — аргумент `--instances`. Готовые инстансы: `mandl`, `mumford0`–`mumford3`,
`macsa_*`, `ekb`.

Правила `Routes.txt`:

- узлы нумеруются с 0;
- строки с `#` и пустые строки пропускаются;
- маршрут записан в одну сторону, обратное направление добавляет код.

При загрузке код проверяет номера узлов, число маршрутов и максимальную длину.
При ошибке запуск останавливается и называет файл.

### Если `Routes.txt` нет

```bash
python scripts/make_init_routes.py
```

Скрипт строит сеть обученной construction-моделью и пишет `Routes.txt` для
`mandl` и `mumford0`–`mumford3`. Флаги: `--instances`, `--seed`, `--instances-dir`,
`--cpu`, `--clean` (только сэмпл модели, без «реалистичной» порчи).
У MACSA и EKB начальная сеть уже есть.

## 3. Рисование

```
python scripts/data_plot.py [ФАЙЛ.pt ...] [--instance ИМЯ] [--out PNG] [--demand] [--ncols N] [--dpi N]
```

Скрипт только рисует PNG. Он ничего не считает.

| Задача | Команда |
|---|---|
| Начальная сеть инстанса | `python scripts/data_plot.py --instance mandl` |
| Результат прогона | `python scripts/data_plot.py artifacts/results/search/search_mandl_ea_seed0_routes.pt` |
| Несколько методов рядом | перечислите несколько `.pt` и задайте `--out` |
| Чужой тензор маршрутов | `python scripts/data_plot.py routes.pt --instance mumford1 --out fig.png` |

- Дамп прогона содержит УДС, сеть `Initial` и сеть каждой точки сетки. Одна
  команда уже даёт сравнение «было → стало».
- При нескольких дампах `Initial` рисуется один раз.
- `--out` по умолчанию — рядом с первым входным файлом.
- `--demand` добавляет панель матрицы корреспонденций. `--ncols` по умолчанию 4,
  `--dpi` — 150.

## 4. Обучение edit-политики

Edit-политика выполняет trim / extend / halt в `nea_edit` и `nea_combined`.

```bash
python scripts/prepare_data.py     # один раз
python scripts/run_training.py     # обучение
```

### 4.1. Датасет

`prepare_data.py` строит случайные графы, получает сеть construction-моделью и
портит её по тирам: `copy_full`, `copy_boundary`, `copy_mixed`, `covered_dup`,
`lc_clean`. Политика учится убирать порчу.

Результат — `datasets/<dataset.dataset_dirname>`, по умолчанию
`lc_copytiers_n1000_n50_r12_len8_15_v1`. Готовый датасет используется повторно.
`--force` строит его заново.

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `dataset_build.n_graphs` | 1000 | число графов |
| `dataset_build.raw_n_nodes` | 50 | узлов в графе |
| `dataset_build.lc_combos` | demand / route / conn | веса цели для начальных сетей |
| `dataset.target_n_routes` | 12 | маршрутов в сети |
| `dataset.min_route_len` / `max_route_len` | 8 / 15 | границы длины |
| `dataset.train_fraction` | 0.9 | доля обучения |

При смене этих ключей смените и `dataset.dataset_dirname`. Иначе новый датасет
перезапишет старый.

```bash
python scripts/prepare_data.py dataset_build.raw_n_nodes=100 dataset.dataset_dirname=lc_copytiers_n1000_n100_r12_len8_15_v1
```

### 4.2. Запуск

Конфиг — `cfg/runs/train.yaml`. Curriculum добавляет тиры порчи по ходу
обучения: сначала `copy_full`, к концу все пять. Валидация — каждые 10 итераций,
жадные действия.

| Флаг | Что делает |
|---|---|
| `--config-name` | другой конфиг под `cfg/` (по умолчанию `runs/train`) |
| `--dry-run` | собирает модель, цель и тренер без датасета и без обучения |
| `--weights-dir` | корень весов |
| `--runs-dir` | рабочая папка прогона: TensorBoard, история |
| `--log-dir` / `--log` | папка лога / путь лога |

Разовые изменения задавайте Hydra-переопределениями:

```bash
python scripts/run_training.py run.name=edit_v2 paths.checkpoint_path=improvement/edit_v2.pt
```

### 4.3. Результат

Чекпоинт — `paths.checkpoint_path` относительно корня весов. По умолчанию
`artifacts/model_weights/improvement/edit_training_v1.pt`.
`paths.init_checkpoint_path` дообучает готовые веса.

Чтобы поиск взял новую модель, укажите её в `cfg/method/nea_edit.yaml` или
`nea_combined.yaml`:

```yaml
models:
  edit:
    checkpoint_path: improvement/edit_v2.pt
```

Другой вариант — оставить имя и задать другой корень: `run_search.py --weights-dir ...`.

## 5. Обучение construction: PPO, decomposed PPO, D3PO

Construction-политика строит сеть с нуля. Алгоритм задаёт группа `trainer`:

```bash
python scripts/run_training.py --config-name runs/train_construction trainer=ppo_construction
python scripts/run_training.py --config-name runs/train_construction trainer=d3po_construction
python scripts/run_training.py --config-name runs/train_construction trainer=decomposed_construction
```

### 5.1. Данные

Один раз сгенерируйте общие файлы:

```bash
python -m connectpt.routes_generator.citygraph_dataset datasets/construction/20_nodes/mixed mixed --min 20 --max 20 -n 32768
python -m connectpt.routes_generator.citygraph_dataset datasets/construction/20_nodes/test mixed --min 20 --max 20 -n 128
```

- Обучающий файл делится 90 / 10 на train и validation.
- Аугментация только у train: масштаб пространства U[0.4, 1.6], спроса U[0.8, 1.2],
  отражение, поворот.
- Test — отдельный файл.
- Все прогоны одного сравнения используют одни и те же файлы.

### 5.2. Настройки по умолчанию

| Параметр | Значение |
|---|---|
| Модель | `construction_holliday`: 13 edge / 11 global / 16 path / 13 halt признаков |
| Batch = minibatch | 64 |
| Horizon | 120 |
| Итерации | 200 |
| γ | PPO 1.0, D3PO 0.95 |
| GAE λ | 0.95 |
| Actor LR / decay | 0.0016134816 / 0.0008404362 |
| Entropy bonus | 0 |
| Сеть | 10 маршрутов по 2–12 остановок |
| Validation | 64 фиксированных города, каждые 10 итераций |
| Pareto | 64 города, 11 предпочтений, greedy + 1 stochastic, каждые 10 итераций |
| Final test | last и best, 21 предпочтение, greedy + 4 stochastic |

- Предпочтения: α = 0 с вероятностью 0.33, α = 1 — 0.33, иначе α ~ U[0, 1].
  Веса держатся до смены graph batch во всех алгоритмах.
- RTT считается в обе стороны.
- Decomposed PPO = D3PO без diversity. Diversity D3PO — полный categorical KL.
- Старые веса с 14 edge-признаками грузятся только с `model=construction`.

### 5.3. Что пишется

Каждый прогон пишет `config.yaml`, `history.csv`, `status.json`, чекпоинты
итераций, `last.pt`, `best.pt`, JSON теста и TensorBoard.

| Тег | Смысл |
|---|---|
| `train/reward` | сумма наград завершённых сетей; NaN без завершений |
| `train/step_reward` | средняя награда шага |
| `train/{cost,att_minutes,rtt_minutes,episode_length}` | метрики завершённых сетей |
| `validation/{cost,att_minutes,rtt_minutes}` | валидация; по `val cost` выбирается best |
| `pareto/{greedy,stochastic}/*` | HV, EU, число различных и недоминируемых сетей |
| `critic/*`, `ppo/*` | loss и explained variance критика, clip fraction, entropy |

HV считается с фиксированной точкой `[3, 3]` в нормированных ATT / RTT. EU —
отрицательная стоимость: больше — лучше.

### 5.4. Очередь сравнения

```powershell
.\scripts\run_construction_comparison.ps1 -ComparisonRoot artifacts/comparisons/construction_seed0 -Seed 0
```

- Запускает алгоритмы по очереди тем же `run_training.py`.
- По умолчанию — 1 536 000 transitions на алгоритм (200 итераций при 64 × 120).
- `-TransitionBudget 6144000` — бюджет upstream (800 итераций).
- `-MaxSecondsPerAlgorithm` — равное время вместо равных transitions.
- `-Algorithms 'ppo,d3po'` — выбор алгоритмов. `-Overrides` — общие переопределения.
- Для каждого seed берите новую папку.

### 5.5. Ночной запуск

```powershell
.\scripts\run_construction_overnight.ps1 -Hours 8 -Seed 0
```

- Запускает PPO и D3PO параллельно двумя обычными процессами `run_training.py`.
- `-Hours 8` — 7 часов обучения на процесс и 1 час на финальную оценку.
- Профиль — `cfg/runs/train_construction_overnight.yaml`: batch 32, horizon 64,
  2 потока на процесс, validation и test по 64 города, 11 предпочтений, 1 stochastic draw.
- Supervisor и пороги памяти не используются.
- `-DryRun` проверяет команды без запуска. `-Wait` держит консоль до конца.
- `launch.json` хранит PID и команды. Каждый процесс пишет свой `status.json`.
- Не уменьшайте batch и horizon без решения Руслана.

## 6. Папки вывода

| Что | Флаг | По умолчанию |
|---|---|---|
| Таблицы и маршруты поиска | `--out-dir` | `artifacts/results/<config>` |
| Рабочая папка прогона, TensorBoard | `--runs-dir` | `artifacts/runs/<run.name>` |
| Логи | `--log-dir` | `artifacts/logs` |
| Веса | `--weights-dir` | `artifacts/model_weights` |

TensorBoard — один сервер на всё: `tensorboard --logdir artifacts`. Отдельный
сервер для прогона не запускайте.

## 7. Фоновый запуск

`run_search.ps1` / `.sh` и `run_training.ps1` / `.sh` принимают те же аргументы.
Процесс отвязан от сессии, лог — в `artifacts/logs/`.

```powershell
.\scripts\run_training.ps1 --config-name runs/train_construction trainer=d3po_construction
.\scripts\run_training.ps1 --wait --dry-run
```

Следить за логом: `Get-Content -Wait -Tail 20 artifacts\logs\<файл>.log`.
Остановить: `Stop-Process -Id <PID>`.
