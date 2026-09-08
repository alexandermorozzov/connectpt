# Запуск

| Скрипт | Что делает |
|---|---|
| `run_nbco.py` | оптимизация сети (NBCO/BCO) по конфигу эксперимента |
| `run_training.py` | обучение edit/trim-модели (PPO) по конфигу обучения |

Запускаются напрямую: `python scripts/run_nbco.py ...`.

### Долгие прогоны: `.sh` и `.ps1`

Если прогон идёт часами и не должен умереть после отключения от сервера или
закрытия окна, запускайте через одноимённый лаунчер вместо `python`:
`run_nbco.sh` / `run_nbco.ps1`, `run_training.sh` / `run_training.ps1`. Они
отвязывают процесс от сессии (Linux — `setsid nohup`, Windows —
`Start-Process`) и печатают PID и путь лога.

**Синтаксис тот же**: все аргументы прокидываются в python-скрипт как есть —
и флаги, и `-p ключ=значение`. Отличие одно, про лог: `.ps1` подставляет
`--log artifacts/logs/nbco_<штамп>.log` (если вы не задали `--log` сами), а
`.sh` дополнительно пишет весь вывод, включая прогресс-бары, в
`artifacts/logs/nbco_<штамп>.log` — рядом с обычным логом прогона.

```bash
./scripts/run_nbco.sh table3_nbco_vs_our --cities Mumford3
```

```powershell
.\scripts\run_nbco.ps1 table3_nbco_vs_our --cities Mumford3
```

Следить и останавливать:

```bash
tail -f artifacts/logs/nbco_<штамп>.log
```

```bash
kill <PID>
```

```powershell
Get-Content -Wait -Tail 20 artifacts\logs\nbco_<штамп>.log
```

```powershell
Stop-Process -Id <PID>
```

---

## 1. `run_nbco.py`

```
python scripts/run_nbco.py <config> [--cities ...] [-p key=value ...]
                           [--out-dir DIR] [--runs-dir DIR] [--weights-dir DIR]
                           [--log-dir DIR] [--log FILE] [--kind K]
```

### Флаги

| Флаг | Что задаёт |
|---|---|
| `<config>` | имя файла из `cfg/experiments/` без `.yaml` (обязателен) |
| `--cities` | список городов, по прогону на город; без флага — город из конфига |
| `-p`, `--param` | параметр эксперимента: `-p ключ=значение` (см. ниже) |
| `--out-dir` | куда класть таблицу, дамп маршрутов и фигуры (`artifacts/results/<config>`) |
| `--runs-dir` | корень рабочих папок прогона: частичный CSV, TensorBoard (`artifacts/runs`) |
| `--weights-dir` | корень, от которого резолвятся чекпоинты (`artifacts/model_weights`) |
| `--log-dir` / `--log` | папка лога / полный путь (`artifacts/logs/<config>_<штамп>.log`) |
| `--kind` | чем рисовать результат: `pareto` (фронт RTT × WMC), `network` (панели маршрутов), `gis` (маршруты на карте). Без флага выбирается сам: гео-данные → `gis`, свип из нескольких точек → `pareto`, иначе `network` |

В скобках — значения по умолчанию. Относительные пути считаются от корня
репозитория, абсолютные берутся как есть.

### Ключи `-p`

`-p` — это флаг, а не параметр: вместо `key=value` подставляется настоящая пара
из таблицы ниже, а сам флаг повторяется столько раз, сколько параметров нужно.

```bash
python scripts/run_nbco.py basic_run -p alpha=0.7 -p n_iterations=300 -p city=Mumford1
```

Каждый ключ по умолчанию берётся из YAML; переданный перекрывает.

| Ключ | Что задаёт |
|---|---|
| `city` | город (то же, что `--cities` с одним значением) |
| `alpha` | вес RTT в RTT–WMC; список = ось сетки (`-p alpha=0,0.5,1`) |
| `adj_target` | целевая степень изменения сети τ; скаляр или список |
| `adj_weight` | вес штрафа за изменение (`0` — штраф выключен) |
| `n_iterations` | итераций BCO на точку сетки |
| `n_routes`, `route_len` | число маршрутов, границы длины (`-p route_len=2,12`) |
| `seed` | сид прогона |
| `cpu` | `true` — считать на CPU |
| `bee_sets`, `models`, `n_bees`, `label` | состав операторов и его подпись (раздел 3) |
| `weights_dir`, `runs_dir` | то же, что одноимённые флаги |

Типы выводятся автоматически: `int` → `float` → `bool` → `None` → список по
запятой → строка. Короткая проверка = маленький бюджет: `-p n_iterations=2`.

---

## 2. Базовый конфиг `basic_run`

`cfg/experiments/basic_run.yaml` — обычный прогон на заданных данных: одна точка
(α=0.5, τ=0.2), один метод (Our NBCO: 5 нейро-rebuild + 5 обученных trim/extend,
10 пчёл), 100 итераций BCO, город Mandl.

```bash
python scripts/run_nbco.py basic_run
```

```bash
python scripts/run_nbco.py basic_run -p city=Mumford1 -p alpha=0.7 -p n_iterations=300 --out-dir artifacts/results/my_run
```

Результат: `<out-dir>/basic_run[_<city>].csv` (метрики), `*_routes.pt` (дамп:
`Initial` + полученная сеть), PNG с маршрутами; частичный CSV и TensorBoard — в
`<runs-dir>/basic_run/`.

### Свои данные

**Стартовая сеть.** По умолчанию строится learned-construction моделью с
«реалистичной» порчей (сид — `run.seed`). Своя задаётся в блоке `data:`
(в файле закомментировано):

```yaml
data:
  source: benchmark
  city: Mandl
  init_dump: artifacts/results/<прогон>/<stem>_routes.pt   # берётся набор с меткой Initial*
  # init_routes_path: datasets/my_routes.pkl               # pickle со списком тензоров маршрутов
```

**Свой город.** Три файла в `datasets/benchmark/`: `<City>Coords.txt` (первая
строка — число узлов, дальше `x y` на узел), `<City>TravelTimes.txt` (квадратная
матрица времён в минутах, `Inf` где нет ребра), `<City>Demand.txt` (квадратная
матрица спроса). Плюс спецификация `cfg/eval/<city>.yaml`:

```yaml
csv: true
n_routes: 12
min_route_len: 2
max_route_len: 15
```

Дальше город обычный: `-p city=<City>`. Геосеть (как EKB) и сценарии MACSA
подключаются своими источниками — `data.source: ekb|macsa`.

### Конфиги экспериментов статьи

Лежат там же, в `connectpt/routes_generator/cfg/experiments/`, и запускаются тем
же скриптом — для воспроизведения результатов:

```bash
python scripts/run_nbco.py table3_nbco_vs_our --cities Mandl Mumford0 Mumford1 Mumford2 Mumford3
```

Что какому артефакту рукописи соответствует — `cfg/experiments/README.md`. Флаги
и ключи `-p` те же; в конфигах с блоком `methods:` состав операторов задан
списком внутри YAML, поэтому `-p bee_sets/models/n_bees/label` для них не
используются.

---

## 3. Параметры операторов (пчёл)

Метод собирается из трёх значений:

| Ключ | Откуда значения | Что задаёт |
|---|---|---|
| `bee_sets` | `cfg/search/bee_sets/*.yaml` | состав колонии: какие пчёлы, сколько, какие действия |
| `models` | `cfg/search/models/*.yaml` | какие модели подняты и с какими весами |
| `n_bees` | число | размер колонии; равен сумме `count` выбранного набора |

```bash
python scripts/run_nbco.py basic_run -p bee_sets=classic_trim_extend -p models=edit_only_seeded -p n_bees=10 -p label="classic + trim/extend"
```

Готовые наборы (Σ — требуемый `n_bees`) и совместимые группы моделей:

| `bee_sets` | Состав | Σ | `models` |
|---|---|---|---|
| `our_nbco` | 5 нейро-rebuild + 5 обученных trim/extend | 10 | `construction_and_edit_seeded` |
| `neural_bco` | 5 нейро-rebuild + 5 эвристических `shorten` | 10 | `construction_only` |
| `classic_bco_paper` | 5 `classic_bco` + 5 `shorten` | 10 | `no_neural_models` |
| `classic_trim_extend` | 5 `classic_bco` + 5 обученных trim/extend | 10 | `edit_only_seeded` |
| `classic_random_trim_extend` | 5 `classic_bco` + 5 случайных trim/extend | 10 | `random_edit_only` |
| `rpc_trim_extend` | 5 RPC + 5 обученных trim/extend | 10 | `edit_only_seeded` |
| `rpc_type2` | 5 RPC + 5 `shorten` | 10 | `no_neural_models` |
| `gnn_random_trim_extend` | 5 нейро-rebuild + 5 случайных trim/extend | 10 | `construction_and_random_edit_seeded` |
| `trim12_extend12` | 12 construction-extend + 12 trim-only | 24 | `construction_and_edit_seeded` |
| `classic_bco` | 20 эвристических `classic_bco` | 20 | `no_neural_models` |
| `nbco_construction` | 20 construction extend/halt | 20 | `construction_only` |
| `nbco_construction_plus_edit` | 10 construction + 10 полного edit | 20 | `construction_and_edit_seeded` |
| `nbco_edit_full` / `nbco_edit_trim` / `nbco_edit_extend` | 20 edit-пчёл: все действия / только trim / только extend | 20 | `edit_only_seeded` |
| `nbco_compound_trim_then_construct` | 20 составных: trim → достроить конструктором | 20 | `construction_and_edit_seeded` |

Группы моделей: `construction_and_edit_seeded` (веса прогонов статьи),
`construction_and_edit`, `construction_only`, `edit_only_seeded` / `edit_only`,
`construction_and_random_edit_seeded`, `random_edit_only`, `no_neural_models`.
Пути чекпоинтов внутри них относительны корню весов (`--weights-dir`).

Правило принятия задаётся не пчелой, а группой
`cfg/search/acceptance/{greedy,mandl_tuned}.yaml`; поля `route_selection` и
`acceptance` внутри наборов пока описательные. Новый состав операторов — новый
файл в `cfg/search/bee_sets/`.

---

## 4. Куда что пишется

| Что | Флаг | По умолчанию |
|---|---|---|
| Таблица, дамп маршрутов, фигуры | `--out-dir` | `artifacts/results/<config>` |
| Частичный CSV, `partial_routes/`, TensorBoard | `--runs-dir` | `artifacts/runs/<run.name>` |
| Лог прогона | `--log-dir` / `--log` | `artifacts/logs/<config>_<штамп>.log` |
| Веса (чтение и сохранение чекпоинтов) | `--weights-dir` | `artifacts/model_weights` |

Те же корни лежат в блоке `paths:` конфига (`cfg/search/bee_colony_base.yaml`,
`cfg/training/*.yaml`) — флаг их перекрывает:

```yaml
paths:
  weights_dir: artifacts/model_weights
  runs_dir: artifacts/runs
  output_dir: ${paths.runs_dir}/${run.name}
```

`run.name` наращивается сегментами `<конфиг>/<город>/<label>`, поэтому разные
города и методы не затирают друг друга. TensorBoard:
`tensorboard --logdir artifacts/runs`.

---

## 5. `run_training.py`

```
python scripts/run_training.py [--config-name C] [--dry-run]
                               [--weights-dir DIR] [--runs-dir DIR]
                               [--log-dir DIR] [--log FILE] [key=value ...]
```

| Флаг | Что задаёт |
|---|---|
| `--config-name` | конфиг из `cfg/` (по умолчанию `training/edit`) |
| `--dry-run` | собрать модель/данные/трейнер и проверить связку, без датасета и PPO |
| `--weights-dir` | корень весов: куда сохранить чекпоинт, откуда взять warm-start |
| `--runs-dir` | корень рабочей папки прогона (TensorBoard, история) |
| `--log-dir` / `--log` | папка лога / полный путь |
| `key=value` | Hydra-override'ы, позиционно, сколько угодно |

Конфиги обучения: `training/edit` (основной), `training/edit_scratch`,
`training/edit_adj_conditioned` (с adjustment-кондиционированием),
`training/edit_scratch_smoke` (нужен датасет `datasets/TEMP_smoke_lc_copytiers`).

Частые override'ы:

| Ключ | Что задаёт |
|---|---|
| `ppo.n_iterations`, `ppo.val_period` | бюджет PPO и период валидации |
| `train_loop.n_iterations` | итерации внешнего цикла + расписание curriculum |
| `paths.checkpoint_path` | имя файла чекпоинта под корнем весов (`improvement/<имя>.pt`) |
| `paths.init_checkpoint_path` | с каких весов стартовать (`null` — со случайных) |
| `data.dataset_dirname` | папка датасета в `datasets/` |
| `adjustment_degree_weight`, `adjustment_degree_target` | шейпинг степени изменения сети в награде |
| `run.seed`, `run.cpu` | сид, принудительный CPU |

```bash
python scripts/run_training.py --dry-run
```

```bash
python scripts/run_training.py --config-name training/edit --weights-dir artifacts/model_weights_v2 paths.checkpoint_path=improvement/edit_v2.pt
```

Обученные веса подключаются к поиску либо флагом `--weights-dir`, либо новой
группой в `cfg/search/models/` с нужным `checkpoint_path`.

---

## 6. Примеры

### Оптимизация (`run_nbco.py`)

Быстрая проверка, что всё собирается и считает — 2 итерации, отдельная папка:

```bash
python scripts/run_nbco.py basic_run -p n_iterations=2 --out-dir artifacts/results/check
```

Рабочий прогон на своём городе и своей точке компромисса, α ближе к времени
в пути, лимит изменения сети 30%:

```bash
python scripts/run_nbco.py basic_run -p city=Mumford1 -p alpha=0.7 -p adj_target=0.3 -p n_iterations=300 --out-dir artifacts/results/mumford1_a07
```

Долгий прогон эксперимента статьи по пяти городам, отсоединённо, со своими
весами:

```bash
./scripts/run_nbco.sh table3_nbco_vs_our --cities Mandl Mumford0 Mumford1 Mumford2 Mumford3 --weights-dir artifacts/model_weights_v2 --out-dir artifacts/results/table3_rerun
```

### Обучение (`run_training.py`)

Проверка, что конфиг собирается (секунды, без датасета и PPO):

```bash
python scripts/run_training.py --dry-run
```

Короткий сквозной прогон во временный чекпоинт — убедиться, что цикл обучения
доходит до сохранения весов:

```bash
python scripts/run_training.py ppo.n_iterations=1 ppo.val_period=1 train_loop.n_iterations=1 paths.checkpoint_path=improvement/TEMP_check.pt
```

Полное обучение отсоединённо, в отдельный корень весов:

```bash
./scripts/run_training.sh --config-name training/edit --weights-dir artifacts/model_weights_v2 paths.checkpoint_path=improvement/edit_v2.pt run.seed=1
```
