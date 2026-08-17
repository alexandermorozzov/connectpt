# Эксперименты ↔ артефакты статьи (MAP)

**Правило (M018): один файл = один артефакт статьи** (таблица/фигура). Имя файла —
по номеру артефакта (`tableN_…` / `figN_…`), а не `batch.yaml`. `\label` рукописи
(`template.tex`) — ключ; номер Table/Figure дублируется здесь для чтения
результатов.

## Три способа задать прогоны артефакта

1. **Скаляры в коде.** Вариативность по alpha / adj_target / route_len / iters —
   один плоский YAML, сетка подставляется `build_experiment(alpha=…, …)`.
   Пример: `table4_fig4_our_pareto` (2D alpha×adj sweep), `ekb_case_study`.
2. **Метод в коде (`methods:`).** Метод отличается только выбором Hydra-групп
   (`bee_sets` + `models`) + `n_bees` + `label` — это НЕ разнородные файлы.
   Один YAML несёт список `methods:`; батч композит его по разу на метод с
   group-override (`build_experiment(bee_sets=…, models=…)`), без листьев-файлов.
   Пример: `table5_fig5_5model` (5 комбинаций), `table3_nbco_vs_our` (2 метода).
3. **Разнородные стеки (`batch.runs:`).** Разный `run.type` / несовместимое
   дерево — только тогда нужна папка + несколько файлов. Пример: бейзлайны
   GA / SA / hyper-heuristics (см. «Бейзлайны» ниже).

## Сопоставление

| Конфиг | `\label` рукописи | № | Способ | Статус |
|---|---|---|---|---|
| `table3_nbco_vs_our` | `tab:nbco-vs-our-only-with-init` | **Table 3** | `methods:` (2) | ✅ M018 |
| `table4_fig4_our_pareto` | `tab:e2c_rtt_median_wmc_mumford0` / `fig:e2-mumford0-ourpareto-topdown` | **Table 4 / Figure 4** | скаляры (alpha×adj) | ✅ M018 (city→Mumford0) |
| `table5_fig5_5model` | `tab:e2-5model-mumford1` / `fig:e2-mumford1-pareto` | **Table 5 / Figure 5** | `methods:` (5) | ✅ M018, цель CLI №2 |
| `ekb_case_study` | (case study) | — | скаляры | ✅ плоский; CLI №1 (adj 0.2 — задача A) |
| `macsa_alpha_sweep` | вход `run_macsa_table_b` → `tab:macsa-tableb` + фигуры | Table 6 (+Figs 6-9) | скаляры (через ф-цию) | ✅ плоский (M018) |
| `macsa_alpha_sweep_iter1` | вход `run_macsa_table_b` → `tab:macsa-alpha-sweep` | Table 7 | скаляры (через ф-цию) | ✅ плоский (M018) |

**Не артефакты статьи (живут в `experiments/`, но по решению пользователя оставлены на месте):**

| Конфиг | Что это | Кто держит |
|---|---|---|
| `bee_type_comparison/` (8) | внутренняя абляция типов пчёл (НЕ Table 2) | тесты + report + `run_experiment_batch.py` |
| `seeded/` (3), `construction_only/mandl` | тест-фикстуры | `test_search_sweep`, `test_seeded_search_run`, `test_declarative` |

**Удалено в M018 (мёртвое, 0 ссылок):** `m0/mumford0/`, `macsa/our_nbco_mandl8`,
`cfg/d3po_50nodes`, `cfg/ppo_mumford3`.

## Бейзлайны (Table 2 `tab:e1u_unified_alpha05_target02`)

Table 2 сравнивает: **Simulated annealing, Genetic algorithm, Hyper-heuristics**,
NeuralBCO, Improved NeuralBCO.

- NeuralBCO / Improved NeuralBCO — bee_colony-методы (способ 2, `methods:`), уже
  покрыты (`neural_bco` / `our_nbco` группы).
- **SA / GA / HH — НЕ в декларативном слое.** Есть только legacy top-level
  `cfg/{sa,ga,hh,nsgaii}_mumford.yaml` + код (`genetic_algorithm.py`,
  `hyperheuristics.py`, `simulated_annealing.py`, `baselines.py`), но **нет
  `run.type`** в `ExperimentRunFactory` (только `bee_colony_search`,
  `edit_training`, `evaluation`, `report`). Значит `experiments/baselines/`
  (способ 3) пока построить нельзя — нужен отдельный шаг: добавить `run.type`
  для SA/GA/HH + `ExperimentRun`-обёртки + декларативные конфиги. Вынесено в
  задачу milestone M018.
