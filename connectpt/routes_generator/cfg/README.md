# cfg/ — иерархия сборки конфигов

Карта верхнего уровня `cfg/`. Секцию статьи (декларативный bee-colony слой)
описывает `experiments/README.md`; здесь — **весь граф сборки** и **легаси-стек B**
(бейзлайны + обучение + evaluation), разведённый по run-type в M019 (задача A).

## Два параллельных фундамента

`cfg/` держит два независимых стека сборки, у каждого свой фундамент:

```
СТЕК A (новый, статья) ─────────────────────────────────────────────
  search/bee_colony_base
     ← objective/rtt_wmc_no_demand + data/mandl_benchmark + acceptance/greedy
     └── experiments/*   (table3/4/5, macsa, ekb, bee_type, seeded)   ✓ упорядочен
                          карта: experiments/README.md

СТЕК B (легаси: бейзлайны / обучение / evaluation) ─────────────────
  experiment/standard   ← cost_function/mine
     ├── baselines/   neural_bco, ga, hh, sa, nsgaii   (+ model/, init/, eval/)
     ├── training/    ppo_20nodes, ppo_50nodes, edit*   (+ model/)
     └── evaluation/  eval_model_mumford, edit_eval     (+ model/)
```

**Фундамент, живущий в корне (не entry-point):**
`bco_mumford.yaml` — алгоритм-конфиг BCO, грузится напрямую
`load_bco_algo_config()` (`objectives/unified.py`) и потребляется **стеком A**
(`search/plan_kwargs.py`, `search/runs.py`) как источник дефолтов пчёл/
worse-accept. Это НЕ baseline — остаётся в корне как foundation.

## Три слоя сборки

| Слой | Что это | Где лежит |
|---|---|---|
| **FOUNDATIONS** | базовые defaults стека | `search/bee_colony_base` (A), `experiment/standard` + `bco_mumford` (B) |
| **GROUPS** | взаимозаменяемые блоки (Hydra-группы) | `objective/`, `data/`, `eval/`, `init/`, `model/`, `model_build/`, `search/{acceptance,bee_sets,models,budgets}`, `experiment/cost_function/` |
| **ENTRY POINTS** | то, что реально запускается | `experiments/` (A); `baselines/` · `training/` · `evaluation/` (B) |

## Стек B — карта сборки (после M019-A)

Каждый entry-point композится **из кода** через `compose(config_name=...)`
(не через `defaults:` другого конфига). Все — с заголовком `# @package _global_`
(обязателен для конфига в подпапке, иначе его ключи вложатся в package по имени
папки); групповые ссылки в их `defaults:` — абсолютные (`/experiment`, `/model`,
`/init`, `/eval`).

| Entry-point | Фундамент + группы (`defaults:`) | Кто композит | Тип |
|---|---|---|---|
| `baselines/neural_bco_mumford` | `/experiment` + `/model:bestsofar_feb2023` + `/init:load` | `utils.py:89,95` | baseline |
| `baselines/ga_mumford` | `/experiment` + `/init:john` + `/eval:mumford0` | `baselines.py:125` (`build_ga_cfg`) | baseline |
| `baselines/hh_mumford` | `/experiment` + `/init:john` + `/eval:mumford0` | `baselines.py:138` (`build_hh_cfg`) | baseline |
| `baselines/sa_mumford` | `/experiment` + `/init:john` + `/eval:mumford0` | `baselines.py:113` (`build_sa_cfg`) | baseline |
| `baselines/nsgaii_mumford` | `/experiment` + `/eval:mumford0` + `override /experiment/cost_function:multi` | `baselines.py:258` | baseline |
| `evaluation/eval_model_mumford` | `/experiment` + `/model:bestsofar_feb2023` | `lc_eval.py:73,100`, `utils.py:465` (default) | evaluation |
| `evaluation/edit_eval` | (см. файл) | `ReportRun` / ноутбук `04_case_study` | evaluation |
| `training/ppo_20nodes` | `/experiment` + `/model:bestsofar_feb2023` | `inductive_route_learning.py:961` (`@hydra.main`) | **training** |
| `training/ppo_50nodes` | `/experiment` + `/model:bestsofar_feb2023` | `edit_bee.py:47`, `validate_edit_models.py:77`, `test_checkpoint_compat.py:30` | **training** |
| `training/edit` | `/experiment` + `/model:bestsofar_feb2023_trim` | база группы (overlay'и ниже) + `training_lc.py:728`, `scripts/train_edit.py` | **training** |
| `training/edit_scratch` | overlay `training/edit` | `load_train_config("edit_scratch")` → ноутбуки 01/02 | **training** |
| `training/edit_scratch_smoke` | overlay `training/edit_scratch` | `load_train_config(...)` (smoke) | **training** |
| `training/edit_adj_conditioned` | overlay `training/edit` | `scripts/train_edit.py --config-name training/edit_adj_conditioned` | **training** |

**Удалён в M019-A:** `bco.yaml` — был мёртв (ноль потребителей, исторический
`build_bco_cfg`-путь снят).

## Проверка: конфиги обучения НЕ потеряны

Все 6 точек входа обучения достижимы после переезда (M019 задача C):

| Конфиг | Точка запуска | Проверено |
|---|---|---|
| `training/ppo_20nodes` | `@hydra.main` — construction-модель | dry-compose ✓ |
| `training/ppo_50nodes` | `compose(...)` — сборка edit-модели | `test_checkpoint_compat` ✓ (старые веса грузятся) |
| `training/edit` | дефолт `scripts/train_edit.py` + база группы | `train_edit.py --dry-run` ✓; `test_training_run` ✓ |
| `training/edit_scratch` | `load_train_config("edit_scratch")` → ноутбуки 01/02 | dry-compose ✓ |
| `training/edit_scratch_smoke` | `load_train_config(...)` (smoke) | dry-compose ✓ |
| `training/edit_adj_conditioned` | `train_edit.py --config-name training/edit_adj_conditioned` | dry-compose ✓ |

Гейт от регресса: `test_connectivity_mode_propagation` требует, чтобы ноутбуки
обучения ходили через `load_train_config(...)` и не инлайнили compose —
переезд/переименование ловится тестами.

## Применённая структура (M019 задача A, 2026-07-14)

```
cfg/
├── search/bee_colony_base            # FOUNDATION (A)
├── experiment/standard               # FOUNDATION (B)   [задача B: → run_base/]
├── bco_mumford.yaml                  # FOUNDATION (B, алго-конфиг BCO)
├── <группы: objective/ data/ eval/ init/ model/ model_build/ search/{...}
│            experiment/cost_function/>   [задача B: cost_function → cost_base/]
└── entry points:
    ├── experiments/   bee_colony_search (стек A)
    ├── baselines/     neural_bco, ga, hh, sa, nsgaii_mumford
    ├── training/      ppo_20nodes, ppo_50nodes, edit{,_scratch,_scratch_smoke,_adj_conditioned}
    └── evaluation/    eval_model_mumford, edit_eval
```

Что потребовалось сверх `git mv` (два Hydra-нюанса):

1. **`# @package _global_`** первой строкой каждого перемещённого конфига —
   иначе конфиг в подпапке кладёт свои ключи в package по имени папки
   (`cfg.baselines.*`), и код-читатели `cfg.batch_size`/`cfg.ppo` ломаются.
2. **Выбор eval-группы стал декларативным.** Append-override `+eval=mumford0`
   резолвится относительно папки primary-конфига (искал `baselines/eval/mumford0`).
   Вместо кода — `- /eval: mumford0` в `defaults:` четырёх baseline-YAML
   (`+eval=mumford0` убран из `baselines.py`). Результат сборки идентичен,
   config-first усилен. (`override`-запись в nsgaii обязана быть последней в
   defaults — `/eval` стоит перед ней.)

Проверки: `pytest -q` 225 passed; dry-compose 12/12 перемещённых конфигов;
`train_edit.py --dry-run` OK.

## Сознательное исключение: `experiment/` и `cost_function/` НЕ переименованы

M019 задача B (`experiment/cost_function/` → `cost_base/`, `experiment/` →
`run_base/`) **отклонена по решению пользователя.** Причина: это не пути файлов,
а **package-имена, читаемые кодом в 46 местах** — `cfg.experiment.cost_function.*`
(cost-ядро: `objectives/factory.py`, `unified.py`; `baselines`, `lc_eval`,
`data/init`, `training_lc`) и `cfg.experiment.{seed,symmetric_routes,logdir}`.
Переезд группы без смены package невозможен, а со сменой правит cost-читающий
код, что нарушает границу «внутреннюю логику cost/BCO/models не трогаем». Выигрыш
— только косметический (де-коллизия имени `experiment/` ↔ `experiments/`), поэтому
фундамент B остаётся `experiment/standard`, cost-группа — `experiment/cost_function/`.

Детали — `board/milestone-019-cfg-build-hierarchy.md`.
