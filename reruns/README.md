# Перезапуск экспериментов статьи (Table 3/4/5 + абляция + EKB)

Тонкие команды поверх CLI-скриптов (`scripts/run_*.py`). Гоняют **те же
декларативные конфиги, что и ноутбуки 03/04**, но пишут все результаты в
**отдельную папку `artifacts/reruns/`** (через suite `suite_rerun`), не трогая
`artifacts/paper_results*`.

## TL;DR — что запускать

| Эксперимент (рукопись) | Linux | Windows (PowerShell) |
|---|---|---|
| **Table 3** — NBCO vs Improved NBCO | `reruns/run_table3.sh` | `.\reruns\run_table3.ps1` |
| **Table 4 / Fig 4** — adj-target × α, Mumford0 | `reruns/run_table4.sh` | `.\reruns\run_table4.ps1` |
| **Table 5 / Fig 5** — 5 комбинаций пчёл, Mumford1 | `reruns/run_table5.sh` | `.\reruns\run_table5.ps1` |
| **Table 5 baselines** — SA/GA/HH с LC-init | `reruns/run_table5_baselines.sh` | `.\reruns\run_table5_baselines.ps1` |
| **Абляция edit-эвристики** — EA / RSL-EA / NEA-Edit | `reruns/run_edit_operator_ablation.sh` | `.\reruns\run_edit_operator_ablation.ps1` |
| **EKB** — кейс Екатеринбурга | `reruns/run_ekb.sh` | `.\reruns\run_ekb.ps1` |

По умолчанию — **полный бюджет** статьи. Результаты → `artifacts/reruns/`.

## 1. Параметры каждого прогона (заданы в YAML, не в команде)

| Эксперимент | `\label` | Граф(ы) | α | adj_target | iter |
|---|---|---|---|---|---|
| Table 3 | `tab:nbco-vs-our-only-with-init` | Mandl, Mumford0-3 | 0, 0.5, 1 | 0.3 | 200 |
| Table 4 / Fig 4 | `tab:e2c_rtt_median_wmc_mumford0` | Mumford0 | 0…1 (шаг .25) | 0.2…1.0 (шаг .2) | 200 |
| Table 5 / Fig 5 | `tab:e2-5model-mumford1` | Mumford1 | 0…1 (шаг .25) | — (adj off) | 200 |
| Edit-оператор | — | Mandl, Mumford0-3 | 0…1 (шаг .1) | — (adj off) | 400 |
| EKB | `tab:ekb-alpha-sweep` | EKB (703 узла) | 0, 0.5, 1 | 0.2 | 50 |

Table 3 — единственный по нескольким городам (скрипт гоняет `suite.cities`
по очереди). Остальные — по одному графу.

## 2. Профили: full и smoke

- **full** (по умолчанию) — полный бюджет статьи, все графы/города, итоговые
  имена файлов без префикса. Это то, что нужно для таблиц/рисунков.
- **smoke** — 1-2 итерации, 1 город (Mandl), префикс `TEMP_`. Только чтобы за
  минуту убедиться, что проводка жива (НЕ научный результат).

```bash
reruns/run_table3.sh              # full
reruns/run_table3.sh smoke        # smoke
reruns/run_table5.sh full --cities Mandl Mumford0 Mumford1 Mumford2 Mumford3
reruns/run_table5_baselines.sh full                                  # eval20k budget
reruns/run_table5_baselines.sh full --budget-mode paper40k           # Holliday EA budget
reruns/run_table5_baselines.sh full --budget-mode scaled             # old per-city budgets
reruns/run_edit_operator_ablation.sh                                 # все benchmark-графы
reruns/run_edit_operator_ablation.sh full --cities Mumford1          # один граф
```
```powershell
.\reruns\run_table3.ps1                 # full
.\reruns\run_table3.ps1 -Profile smoke  # smoke
.\reruns\run_table5.ps1 -Profile full -Cities Mandl,Mumford0,Mumford1,Mumford2,Mumford3
.\reruns\run_table5_baselines.ps1 -Profile full                                  # eval20k budget
.\reruns\run_table5_baselines.ps1 -Profile full -BudgetMode paper40k             # Holliday EA budget
.\reruns\run_table5_baselines.ps1 -Profile full -BudgetMode scaled               # old per-city budgets
.\reruns\run_edit_operator_ablation.ps1                                           # все benchmark-графы
.\reruns\run_edit_operator_ablation.ps1 -Profile full -Cities Mumford1            # один граф
```

Linux-скрипты запускают прогон **detached** (`setsid nohup`) — переживает обрыв
SSH; печатают PID, путь к логу, TB-подсказку и `kill`-команду. Windows-скрипты
стримят вывод в консоль и пишут лог в файл.

Для Table 5 baselines дефолтный full-бюджет — `eval20k`: SA/HH получают
`n_iterations=20000`, GA получает `population_size=10`, `n_iterations=1000`.
Это примерно соответствует Table 5 BCO: `200 * 10 * 5 * 2 = 20000`
candidate evaluations.

## 3. Куда что пишется

### `artifacts/reruns/` — делверблы (для таблиц и рисунков)
| Файл | Что |
|---|---|
| `<exp_name>.csv` | таблица метрик: RTT, WMC, ATT, adj_vs_seed, redun%, cost по α (Table 3 — отдельный CSV на город) |
| `<exp_name>_pareto.png` | Парето-фигура (Fig 4 / Fig 5); EKB — маршруты на гео-подложке |
| `<exp_name>_routes.pt` | дамп маршрутов лучшего решения (одиночные прогоны: Table 4, EKB) |
| `<exp_name>_run.log` | INFO-лог прогона |

Имена (`exp_name`): Table 3 — `table3_nbco_vs_our_<city>`; Table 4 —
`table4_fig4_our_pareto_mumford0`; Table 5 — `table5_fig5_5model_mumford1`;
EKB — `ekb_our_nbco`.

### `artifacts/runs/<experiment>/…` — по ходу прогона (устойчивость + история)
| Файл | Что |
|---|---|
| `…_partial.csv` | дозапись после каждой точки α (креш не теряет посчитанное) |
| `partial_routes/<α>.pt` | маршруты каждой α — для route-diff рисунков |
| `tb/<α>/events.*` | TensorBoard: история best-cost по BCO-итерациям |

Корни `artifacts/runs/<experiment>`: `table3_nbco_vs_our`,
`table4_fig4_our_pareto`, `table5_fig5_5model_mumford1`, `ekb_case_study`.

### `artifacts/cli_logs/rerun_<exp>.log`
Лог запуска команды. Linux — полный (stdout+stderr, из `nohup`); Windows —
stderr/tqdm (stdout идёт в консоль). Полный INFO-лог всегда есть и в
`artifacts/reruns/<exp_name>_run.log`.

## 4. Мониторинг онлайн

```bash
# лог прогона (Linux)
tail -f artifacts/cli_logs/rerun_table3.log

# TensorBoard — история сходимости
tensorboard --logdir artifacts/runs/table3_nbco_vs_our
tensorboard --logdir artifacts/runs/table4_fig4_our_pareto
tensorboard --logdir artifacts/runs/table5_fig5_5model_mumford1
tensorboard --logdir artifacts/runs/ekb_case_study/tb
```

В TensorBoard кривые — это теги с префиксом **`best `** (`best cost`,
`best RTT`, `best median_connectivity_weighted`, …): значение метрики по
BCO-итерациям, один run на (метод × α) / (α × adj_target). Итоговые числа
берутся из CSV.

## 5. Прямой вызов (если нужен нестандартный флаг)

```bash
.venv/Scripts/python.exe scripts/run_table3.py --profile full --suite suite_rerun --n-iterations 500
.venv/Scripts/python.exe scripts/run_table3.py --profile full --suite suite_rerun --cities Mandl Mumford0
```
Флаги: `--profile {smoke,full}`, `--suite <name>` (full-suite), `--suite-smoke <name>`,
`--n-iterations N`, `--cities …` (только Table 3).

## 6. Данные и веса

- Бенчмарки (Mandl, Mumford0-3) — `datasets/benchmark/`.
- EKB — `datasets/EKB/`.
- Веса моделей — `artifacts/model_weights*/` (грузятся как есть).

## 7. Заметки

- **Итерации Table 3 — 200** (как в статье). Больше — флагом `--n-iterations`.
- Полные прогоны идут часами (EKB на 703 узла и Mumford2/3 — дольше всех).
- `artifacts/` в `.gitignore` — результаты локальные, в репозиторий не попадают.
- Один прогон никогда не затирает `artifacts/paper_results*` (пишем в `reruns/`).
