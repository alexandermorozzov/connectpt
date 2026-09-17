# Запуск

| Скрипт | Что делает |
|---|---|
| `run_search.py` | оптимизация сети: EA / NEA / NEA-Edit / NEA-Combined |
| `make_init_routes.py` | один раз строит начальные сети для бенчмарков |
| `data_plot.py` | рисует маршруты поверх УДС |
| `prepare_data.py` | генерирует обучающий датасет |
| `run_training.py` | обучает edit-политику (PPO) |

Долгие прогоны — через `run_search.sh` / `run_search.ps1` (и такие же для обучения): те же
аргументы, но процесс отвязан от сессии (`setsid nohup` / `Start-Process`), лог в
`artifacts/logs/`.

---

## 0. Где что лежит

```
connectpt/routes_generator/cfg/
  runs/            ЧТО ЗАПУСКАЕТСЯ — здесь собраны кубики
    search.yaml       прогон EA / NEA / NEA-Edit / NEA-Combined
    prepare_dataset.yaml генерация обучающего датасета
    train.yaml        обучение edit-политики

  method/          КУБИК: ea, nea, nea_edit, nea_combined
  cost/            КУБИК: цель оптимизации
  dataset/         КУБИК: обучающий датасет — геометрия, генерация, тиры
  model/           КУБИК: архитектуры нейросетей
  instance/        КУБИК: инстансы — путь, число и длина маршрутов
  experiment/      общие настройки: seed, symmetric_routes
```

Файл в `runs/` перечисляет в `defaults`, какие кубики берёт, и добавляет своё —
инстанс, сетку, бюджет:

```yaml
# cfg/runs/search.yaml
defaults:
  - /cost: rtt_wmc          # ← кубик из cost/
  - /adjustment: cap        # ← вариант ограничения adjustment degree
  - /instance: mandl        # ← кубик из instance/
  - /method: nea_combined   # ← кубик из method/, подменяется флагом --methods
  - _self_

sweep: {alpha: [0, 0.25, 0.5, 0.75, 1], adj_weight: 0.0, n_iterations: 400}
```

Чтобы поменять метод или цель — подставить другое имя кубика. Чтобы поменять
город — положить папку в `datasets/instances/` и файл в `cfg/instance/` (см. §2).

---

## 1. Оптимизация

```
python scripts/run_search.py [--methods ...] [--instances ...] [--alpha ...] [--seeds ...]
                         [--iters N] [--adj-weight W] [--adj-target T]
                         [--out-dir DIR] [--runs-dir DIR] [--weights-dir DIR]
                         [--log-dir DIR] [--cpu] [--no-initial]
```

Без аргументов воспроизводит валидацию из статьи: 4 метода × 5 бенчмарков ×
α ∈ {0, 0.25, 0.5, 0.75, 1} × 5 сидов, 400 итераций, штраф за изменение сети
выключен.

| Флаг | Значение по умолчанию |
|---|---|
| `--methods` | `ea nea nea_edit nea_combined` |
| `--instances` | `mandl mumford0 mumford1 mumford2 mumford3` |
| `--alpha` | `0 0.25 0.5 0.75 1` |
| `--seeds` | `0 1 2 3 4` |
| `--iters` | из конфига (400) |
| `--adj-weight` | из конфига (0) |
| `--out-dir` | `artifacts/results/validation` |
| `--runs-dir` | `artifacts/runs` |
| `--weights-dir` | `artifacts/model_weights` |
| `--log-dir` | `artifacts/logs` |

На выходе на каждый инстанс: `validation_<name>.csv` (строка на точку сетки,
плюс строки `Initial`) и дампы маршрутов
`validation_<name>_<method>_seed<N>_routes.pt`.

Примеры:

```bash
python scripts/run_search.py --methods nea_edit --instances mandl --out-dir artifacts/results/nea_edit_mandl
```

```bash
python scripts/run_search.py --alpha 0 0.5 1 --seeds 0 --instances mumford1 mumford2
```

Со штрафом за изменение сети — `--adj-weight` включает его, `--adj-target`
задаёт бюджет (доля, а не проценты):

```bash
python scripts/run_search.py --instances macsa_mandl_8 --adj-weight 10 --adj-target 0.0602 --out-dir artifacts/results/macsa_mandl_8
```

### Методы

| `--methods` | Первый оператор | Второй оператор | Модели |
|---|---|---|---|
| `ea` | замена маршрута кратчайшим путём | локальная правка концов (p=0.2) | — |
| `nea` | нейросетевая перестройка | локальная правка концов | construction |
| `nea_edit` | замена кратчайшим путём | обученный trim/extend | edit |
| `nea_combined` | нейросетевая перестройка | обученный trim/extend | construction + edit |

Файлы: `connectpt/routes_generator/cfg/method/{ea,nea,nea_edit,nea_combined}.yaml`
— в каждом состав пчёл и профили моделей. Прогон целиком: `cfg/runs/search.yaml`
(инстанс, сетка α, бюджет, расписание). Цель: `cfg/cost/rtt_wmc.yaml`.

## 2. Данные: один инстанс — одна папка

```
datasets/instances/<name>/
    Coords.txt        первая строка — число узлов N, дальше N строк «x y»
    TravelTimes.txt   матрица N×N, В МИНУТАХ (код переводит в секунды)
    Demand.txt        матрица N×N корреспонденций
    Routes.txt        начальная сеть: строка = маршрут, узлы через пробел
```

Плюс `connectpt/routes_generator/cfg/instance/<name>.yaml` с путём и ограничениями:

```yaml
n_routes: 6
min_route_len: 2
max_route_len: 8
```

Чтобы добавить свой город, достаточно положить папку и этот файл: имя папки
и есть аргумент `--instances`.

`Routes.txt` устроен так:

```
# mandl initial network -- learned construction + realistic tier, seed 0
# 0-indexed; one route per line
0 1 2 5 7 9 12
6 14 5 3 4
```

Узлы 0-индексированные. Строки, начинающиеся с `#`, и пустые пропускаются.
Паддинга нет, маршруты разной длины. Маршрут односторонний — обратное
направление добавляет код.

При загрузке проверяются диапазон узлов, число маршрутов против `n_routes` и
длина самого длинного маршрута против `max_route_len`; расхождение прерывает
запуск с указанием файла и ожидаемого значения.

### Если начальной сети нет

```bash
python scripts/make_init_routes.py
```

Строит её обученной construction-моделью и пишет `Routes.txt` в папку инстанса
(по умолчанию `mandl` и `mumford0-3`). Флаги: `--instances`, `--seed`,
`--clean` (только сэмпл модели, без «реалистичной» порчи), `--instances-dir`,
`--cpu`. Инстансы со своей начальной сетью (MACSA, EKB) в этом не нуждаются.

## 3. Рисование

Рисует наборы маршрутов поверх УДС и сохраняет PNG. Ничего не считает и не
оптимизирует.

```
python scripts/data_plot.py [ФАЙЛ.pt ...] [--instance NAME] [--out PNG]
                            [--demand] [--ncols N] [--dpi N]
```

Маршруты берутся из одного из двух источников.

**Из папки инстанса** — то есть из `Routes.txt`. Файл указывать не надо, хватает
имени:

```bash
python scripts/data_plot.py --instance mandl
```

Читает `datasets/instances/mandl/`, рисует её начальную сеть и кладёт PNG рядом
с `Routes.txt`.

**Из артефактов прогона** — дамп `.pt`, который пишет `run_search.py`. УДС лежит
внутри дампа, так что `--instance` не нужен:

```bash
python scripts/data_plot.py artifacts/results/validation/validation_mandl_ea_seed0_routes.pt
```

### Было → стало

Отдельно совмещать ничего не надо: **дамп прогона уже содержит и то и другое**.
В нём лежит набор `Initial` — побайтово тот же, что `Routes.txt` инстанса, — и
по набору на каждую точку сетки:

```
Initial | EA a=0.0 t=None | EA a=0.5 t=None | EA a=1.0 t=None
```

Все они рисуются панелями в одной сетке, поэтому одна команда выше уже даёт
сравнение начальной сети с оптимизированными. Ширина сетки — `--ncols`.

### Сравнение методов

Дамп пишется на каждый метод отдельно. Чтобы увидеть их рядом, перечисли файлы
через пробел — наборы сливаются в одну картинку, `Initial` остаётся в одном
экземпляре:

```bash
python scripts/data_plot.py \
    artifacts/results/validation/validation_mandl_ea_seed0_routes.pt \
    artifacts/results/validation/validation_mandl_nea_edit_seed0_routes.pt \
    --out compare.png --ncols 5
```

Получится пять панелей: `Initial`, `EA a=0.0`, `EA a=1.0`, `NEA-Edit a=0.0`,
`NEA-Edit a=1.0`. Ярлыки уже содержат имя метода, так что путать нечего.

### Чужой файл маршрутов

Если это голый тензор без УДС, нужен `--instance` — откуда взять координаты и
рёбра:

```bash
python scripts/data_plot.py routes.pt --instance mumford1 --out fig.png
```

| Флаг | Что делает |
|---|---|
| `--instance` | откуда брать УДС; без файлов маршрутов — ещё и что рисовать |
| `--out` | путь PNG (по умолчанию рядом с первым входным файлом, то же имя) |
| `--demand` | добавить панель матрицы корреспонденций |
| `--ncols` | колонок в сетке панелей (по умолчанию 4) |
| `--dpi` | разрешение (по умолчанию 150) |

## 4. Обучение

Обучается **edit-политика** — та, что делает trim/extend/halt в `nea_edit` и
`nea_combined`. Два шага: собрать датасет, затем запустить PPO.

```bash
python scripts/prepare_data.py     # один раз
python scripts/run_training.py     # обучение
```

### 4.1 Датасет

Свои данные для обучения не нужны: `prepare_data.py` генерирует их сам.
Он строит 1000 случайных графов на 50 узлов, для каждого получает сеть маршрутов
обученной construction-моделью и портит её по четырём «тирам» (дублирование
участков, детуры, потеря покрытия). Политика учится эту порчу вычищать.

Кладётся всё в `datasets/<dataset.dataset_dirname>` — по умолчанию
`lc_copytiers_n1000_n50_r12_len8_15_v1`. Готовый датасет переиспользуется;
`--force` перегенерирует.

```bash
python scripts/prepare_data.py --force
python scripts/prepare_data.py dataset_build.raw_n_nodes=100 dataset.dataset_dirname=lc_copytiers_n1000_n100_r12_len8_15_v1
```

Позиционные аргументы — Hydra-переопределения того же конфига. Что там можно
крутить (`cfg/dataset/lc_copytiers.yaml`):

| Ключ | Значение по умолчанию | Что задаёт |
|---|---|---|
| `dataset_build.n_graphs` | 1000 | сколько графов сгенерировать |
| `dataset_build.raw_n_nodes` | 50 | узлов в каждом графе |
| `dataset_build.lc_combos` | demand / route / conn | с какими весами цели строить сети |
| `dataset.target_n_routes` | 12 | маршрутов в сети |
| `dataset.min_route_len` / `max_route_len` | 8 / 15 | границы длины |
| `dataset.train_fraction` | 0.9 | доля на обучение, остальное на валидацию |

Меняешь эти значения — меняй и `dataset.dataset_dirname`, иначе новый датасет
ляжет поверх старого.

### 4.2 Запуск обучения

```bash
python scripts/run_training.py
```

Всё берётся из `cfg/runs/train.yaml`. Два вложенных цикла: внешний
`train_loop.n_iterations: 700` и PPO `ppo.n_iterations: 200`, валидация каждые
`ppo.val_period: 10` итераций детерминированными действиями. Действие `halt`
замаскировано на первом шаге первые 150 итераций (`force_nonhalt_first_step_until_iter`),
чтобы политика не научилась сразу прекращать правку. Curriculum добавляет тиры порчи по мере обучения:
сначала только `copy_full`, к концу все пять.

Быстрая проверка, что всё собирается — без датасета и без PPO-цикла:

```bash
python scripts/run_training.py --dry-run
```

| Флаг | Что делает |
|---|---|
| `--config-name` | другой конфиг под `cfg/` (по умолчанию `runs/train`) |
| `--dry-run` | собрать модель, цель и трейнер, не грузя датасет |
| `--weights-dir` | корень, относительно которого лежат чекпоинты |
| `--runs-dir` | рабочая папка прогона: TensorBoard, промежуточные файлы |
| `--log-dir` / `--log` | папка лога / явный путь лога |

Разовые изменения удобно делать Hydra-переопределениями, не трогая файл:

```bash
python scripts/run_training.py run.name=edit_v2 paths.checkpoint_path=improvement/edit_v2.pt
```

### 4.3 Куда попадает результат и как его подключить

Чекпоинт пишется в `paths.checkpoint_path` **относительно корня весов**:
по умолчанию `artifacts/model_weights/improvement/edit_training_v1.pt`.
`paths.init_checkpoint_path` позволяет дообучать с уже готовых весов.

Чтобы поиск начал использовать обученную модель, укажи её в файле метода —
`cfg/method/nea_edit.yaml` или `nea_combined.yaml`:

```yaml
models:
  edit:
    checkpoint_path: improvement/edit_v2.pt
```

Либо оставь имя как есть и подставь другой корень: `run_search.py --weights-dir ...`.

Ход обучения смотрится в TensorBoard:

```bash
tensorboard --logdir artifacts/runs
```

Долгий прогон лучше отвязать от сессии — `run_training.sh` / `run_training.ps1`
принимают те же аргументы и пишут лог в `artifacts/logs/`.

## 5. Куда что пишется

| Что | Флаг | По умолчанию |
|---|---|---|
| Таблицы и дампы маршрутов | `--out-dir` | `artifacts/results/validation` |
| Частичный CSV, TensorBoard | `--runs-dir` | `artifacts/runs/<run.name>` |
| Логи | `--log-dir` | `artifacts/logs` |
| Веса | `--weights-dir` | `artifacts/model_weights` |

TensorBoard: `tensorboard --logdir artifacts/runs`.
