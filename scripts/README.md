# Запуск

Все команды выполняйте из корня репозитория.

| Скрипт | Назначение |
|---|---|
| `run_search.py` | Оптимизация сети: EA, NEA, NEA-Edit, NEA-Combined. |
| `run.py` | Один конфиг `cfg/runs/<имя>.yaml`: датасет edit, обучение, сравнение. |
| `make_init_routes.py` | Строит начальные сети бенчмарков: `--generator lc`, `john`, `nikolic`. Группы конфига — `cfg/runs/init_routes.yaml`. |
| `launch.ps1` / `launch.sh` | Любой из скриптов выше в фоне, с логом. |

Конфиги лежат в `connectpt/routes_generator/cfg/`. Точки входа — `cfg/runs/*.yaml`.
Устройство конфигов — `cfg/README.md`.

## 1. Поиск

```
python scripts/run_search.py [--methods ...] [--instances ...] [--alpha ...] [--seeds ...]
                             [--iters N] [--adj-weight W] [--adj-target T ...]
                             [--out-dir DIR] [--runs-dir DIR] [--weights-dir DIR]
                             [--log-dir DIR] [--log FILE] [--cpu] [--no-initial] [--dry-run]
```

Без аргументов скрипт делает один короткий прогон. Один запуск `SearchRun` считает одну
точку (метод, инстанс, seed, α, цель adjustment); перебор сетки делает скрипт.

| Флаг | По умолчанию |
|---|---|
| `--methods` | `nea_combined` |
| `--instances` | `mandl` |
| `--alpha` | `0.5` |
| `--seeds` | `0` |
| `--iters` | из `cfg/sweep/single.yaml` (400) |
| `--adj-weight`, `--adj-target` | из `cfg/adjustment/cap.yaml` (вес 10, цель 0.2) |
| `--out-dir` | `artifacts/results/search` |

`--adj-weight 0` выключает штраф за изменение сети. `--adj-target` — доля, а не проценты;
можно задать несколько значений: сетка α × цель.

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

Состав пчёл и модели метода — `cfg/search/method/<метод>.yaml`. Цель — `cfg/objectives/cost/rtt_wmc.yaml`.
Группы `method.bees` исполняются по порядку записи. `allowed_actions` задаёт
всю маску нейронного edit-оператора. Общие настройки колонии также лежат в `method`.

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
`--cpu`, `--generator {lc,john,nikolic}`, `--clean` (LC без «реалистичной» порчи).
У MACSA и EKB начальная сеть уже есть.

## 3. Визуализация

Графики строятся в ноутбуках `examples/`. Дамп поиска содержит город, сеть
`Initial` и сети точек поиска. Загрузка — `run.artifacts.load_routes`, метрики —
`objectives.metrics.evaluate_network`. В пакете и логгере рисования нет.

## 4. Обучение edit-политики

Edit-политика выполняет trim / extend / halt в `nea_edit` и `nea_combined`.

```bash
python scripts/generate_dataset.py --config-name runs/generate_edit --out datasets/lc_copytiers_manifest_v1
python scripts/run.py --config-name runs/train dataset.path=datasets/lc_copytiers_manifest_v1
```

### 4.1. Датасет

`runs/generate_edit` строит случайные графы, получает сеть construction-моделью и
портит её по тирам: `copy_full`, `copy_boundary`, `copy_mixed`, `covered_dup`,
`lc_clean`. Политика учится убирать порчу.

Результат записывается в обязательную папку `--out`: `dataset.pt` и `manifest.json`.
Повтор с теми же параметрами и входными весами пропускается. Неполные данные или
другие параметры вызывают ошибку; `--force` явно пересобирает эту папку.
`--dry-run` проверяет конфиг и входы без записи данных.

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `dataset_build.n_graphs` | 1000 | число графов |
| `dataset_build.source.min_nodes` | 50 | узлов в графе; max_nodes и fixed_nodes берутся из этого значения |
| `dataset_build.lc_combos` | demand / route / conn | веса цели для начальных сетей |
| `dataset_build.limits.n_routes` | 12 | маршрутов в сети |
| `dataset_build.limits.min_route_len` / `max_route_len` | 8 / 15 | границы длины |
| `dataset_build.split.train_fraction` | 0.9 | доля обучения |

Обучение читает путь `dataset.path`, размеры и готовое разбиение — из manifest.
Старый `lc_copytiers_n1000_n50_r12_len8_15_v1` также читается; для него конфиг задаёт
`dataset.legacy_limits` и `dataset.legacy_split`.

```bash
python scripts/generate_dataset.py --config-name runs/generate_edit --out datasets/lc_copytiers_n100_manifest_v1 dataset_build.source.min_nodes=100
python scripts/generate_dataset.py --config-name runs/generate_edit --out datasets/lc_copytiers_converted --convert-from datasets/lc_copytiers_n1000_n50_r12_len8_15_v1
```

### 4.2. Запуск

Конфиг — `cfg/runs/train.yaml`. Curriculum добавляет тиры порчи по ходу
обучения: сначала `copy_full`, к концу все пять. Валидация — каждые 10 итераций,
жадные действия.
Curriculum меняет только train; фиксированный состав validation записан в manifest.
`curriculum.enabled=false` отключает расписание. Итоговая edit-оценка записывается
как `eval/final_validation/…`; отдельный test не создаётся.

| Флаг | Что делает |
|---|---|
| `--config-name` | конфиг под `cfg/`, обязателен |
| `--dry-run` | собирает объекты и проверяет ключи конфига, без датасета и обучения |
| `--weights-dir` | корень весов |
| `--runs-dir` | рабочая папка прогона: TensorBoard, история |
| `--log-dir` / `--log` | папка лога / путь лога |

Разовые изменения задавайте Hydra-переопределениями:

```bash
python scripts/run.py --config-name runs/train run.name=edit_v2 paths.checkpoint_path=improvement/edit_v2.pt
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

## 5. Обучение construction: PPO и D3PO

Construction-политика строит сеть с нуля. Алгоритм задаёт группа `algorithm`:

```bash
python scripts/run.py --config-name runs/train_construction training/algorithm=ppo
python scripts/run.py --config-name runs/train_construction training/algorithm=d3po
python scripts/run.py --config-name runs/train_construction training/algorithm=d3po algorithm.diversity_weight=0
```

### 5.1. Данные

Один раз сгенерируйте общие файлы:

```bash
python scripts/generate_dataset.py --config-name runs/generate_construction --out datasets/construction/20_nodes/mixed_manifest
python scripts/generate_dataset.py --config-name runs/generate_construction --out datasets/construction/20_nodes/test_manifest dataset_build.n_graphs=128 run.seed=1
python scripts/run.py --config-name runs/train_construction dataset.path=datasets/construction/20_nodes/mixed_manifest dataset.test_path=datasets/construction/20_nodes/test_manifest
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
- Абляция без diversity: `training/algorithm=d3po algorithm.diversity_weight=0`.
  Diversity D3PO — полный categorical KL.
- Старые веса с 14 edge-признаками грузятся только с `policy/model=construction`.

### 5.3. Что пишется

Папка прогона `<runs-dir>/<run.name>/`:

| Файл | Содержимое |
|---|---|
| `run.json` | commit, seed, argv, status; evaluation_protocol: split, индексы графов, seed, сетки, reference и бюджеты |
| `config.yaml` | собранный конфиг |
| `history.csv` | строка на шаг, столбец на метрику (те же теги, что в TensorBoard) |
| `tensorboard/` | события TensorBoard |
| `critic_snapshot.pt` | параметры критика, последние значения и targets |

Кроме того: чекпоинты итераций, `last.pt`, `best.pt` (в корне весов).

| Тег | Смысл |
|---|---|
| `rollout/ep_rew_mean`, `rollout/ep_len_mean` | завершённые сети; NaN без завершений |
| `rollout/step_reward` | средняя награда шага |
| `rollout/{cost,att_minutes,rtt_minutes}` | метрики завершённых сетей |
| `eval/cost`, `eval/objective/*`, `eval/penalty/*`, `eval/monitor/*` | валидация; по cost выбирается best |
| `pareto/{greedy,stochastic}/*` | HV, EU, число различных и недоминируемых сетей |
| `eval/test/{last,best}/{greedy,stochastic}/*` | итоговый cost, цели, штрафы, мониторинг; Pareto при его включении |
| `train/{value_loss,clip_fraction,entropy_loss,approx_kl}` | критик и обновление политики |
| `time/total_timesteps` | число переходов среды |

HV считается с фиксированной точкой `[3, 3]` по активным нормированным целям.
Их порядок указан в протоколе оценки: ATT/RTT для construction, RTT/WMC для edit.
EU — отрицательная стоимость: больше — лучше. Таблица тегов — `docs/config-keys.md`.
Пакет пишет скаляры и данные; визуализацию делают ноутбуки `examples/`.

### 5.4. Сравнение алгоритмов

Все прогоны одного сравнения лежат в `<папка>/runs/<имя>`: задайте `run.name` и
`--runs-dir`. Равный бюджет задайте одинаковыми переопределениями.

```powershell
$cmp = 'artifacts/comparisons/construction_seed0'
foreach ($a in 'ppo', 'd3po') {
    .\scripts\launch.ps1 --wait run.py --config-name runs/train_construction training/algorithm=$a `
        run.name=$a run.seed=0 --runs-dir $cmp/runs --weights-dir $cmp/weights
}
```

- Сопоставляйте скаляры прогонов в общем TensorBoard и `history.csv`.
- Для сравнения задайте одинаковые seed, commit, данные, split, протокол оценки
  и бюджет: transitions или `train_loop.max_seconds`.
- `evaluation.pareto_enabled=true` подключает HV / EU; construction включает его
  по умолчанию, edit выключает. Выключенный монитор не выполняет дополнительные
  оценки сетки. Итоговый test cost записывается и без Pareto.
- Для каждого seed берите новую папку.

### 5.5. Ночной запуск

Два обычных процесса, PPO и D3PO параллельно:

```powershell
$cmp = 'artifacts/comparisons/construction_overnight_seed0'
.\scripts\launch.ps1 run.py --config-name runs/train_construction_overnight training/algorithm=ppo run.name=ppo --runs-dir $cmp/runs --weights-dir $cmp/weights
.\scripts\launch.ps1 run.py --config-name runs/train_construction_overnight training/algorithm=d3po run.name=d3po --runs-dir $cmp/runs --weights-dir $cmp/weights
```

- Профиль — `cfg/runs/train_construction_overnight.yaml`: batch 32, horizon 64,
  2 потока на процесс, validation и test по 64 города, 11 предпочтений,
  1 stochastic draw, `train_loop.max_seconds: 25200` (7 часов обучения).
- Supervisor и пороги памяти не используются.
- Перед запуском проверьте команды: `--wait ... --dry-run`.
- Утром: сопоставьте теги `eval/test/best/*` в общем TensorBoard;
  проверьте протоколы в `run.json` и бюджеты в `config.yaml`.
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

`launch.ps1` (Windows) и `launch.sh` (Linux) запускают любой скрипт из `scripts/`
в фоне. Первый аргумент — скрипт, остальные уходят ему без изменений.

```powershell
.\scripts\launch.ps1 run.py --config-name runs/train_construction training/algorithm=d3po
.\scripts\launch.ps1 run_search.py --methods ea nea --instances mandl --alpha 0 1
.\scripts\launch.ps1 --wait run.py --config-name runs/train --dry-run
```

- Без `--log` лаунчер задаёт `--log <log-dir>/<скрипт>_<время>.log`
  (`<log-dir>` — `--log-dir` или `artifacts/logs`).
- Консольный вывод (tqdm, трассировки) — рядом с логом: `<лог>.out`, `<лог>.err`
  (на Linux — один `<лог>.out`).
- `--wait` ждёт конца и падает при ненулевом коде выхода.

Следить за логом: `Get-Content -Wait -Tail 20 <лог>` или `tail -f <лог>`.
Остановить: `Stop-Process -Id <PID>` или `kill <PID>`.
