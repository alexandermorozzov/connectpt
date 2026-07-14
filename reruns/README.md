# Перезапуски экспериментов статьи (Table 3/4/5 + EKB)

Тонкие команды поверх CLI-скриптов (`scripts/run_*.py`) — гоняют те же
декларативные конфиги, что и ноутбуки 03/04, но пишут **все делверблы в
отдельную папку `artifacts/reruns/`** (через suite `suite_rerun`), не трогая
`artifacts/paper_results*`.

## Команды

| Эксперимент | Linux | Windows |
|---|---|---|
| **Table 3** (NBCO vs Improved NBCO, Mandl+Mumford0-3, α∈{0,0.5,1}, τ=0.3, 200 iter) | `reruns/run_table3.sh` | `.\reruns\run_table3.ps1` |
| **Table 4 / Fig 4** (adj-target×α свип, Mumford0, 200 iter) | `reruns/run_table4.sh` | `.\reruns\run_table4.ps1` |
| **Table 5 / Fig 5** (5 комбинаций пчёл, Mumford1, adj off) | `reruns/run_table5.sh` | `.\reruns\run_table5.ps1` |
| **EKB** (α∈{0,0.5,1}, τ=0.2, 25 iter, 703 узла) | `reruns/run_ekb.sh` | `.\reruns\run_ekb.ps1` |

Полный бюджет — по умолчанию. Быстрая проверка проводки (1 итерация, 1 город,
префикс `TEMP_`): `reruns/run_table3.sh smoke` / `.\reruns\run_table3.ps1 -Profile smoke`.

Linux-скрипты запускают процесс **detached** (`setsid nohup`) — переживает обрыв
SSH; печатают PID, путь к логу и TB-подсказку. Windows-скрипты стримят лог в
консоль и в файл (`Tee-Object`).

## Что пишется (для таблиц и рисунков)

В `artifacts/reruns/` (делверблы):
- `<stem>.csv` — таблица метрик (RTT/WMC/ATT/adj/redun%/cost по α, для Table 3 — по городам);
- `<stem>_pareto.png` — Парето-фигура (Fig 4 / Fig 5), где применимо;
- `<stem>_routes.pt` — дамп маршрутов (одиночные прогоны: Table 4, EKB);
- `<stem>_run.log` — лог прогона.

В `artifacts/runs/<experiment>/…` (по ходу прогона, устойчивость к падению):
- `…_partial.csv` + `partial_routes/<α>.pt` — инкрементально после каждой точки
  (креш не теряет посчитанное; маршруты каждой α — для route-diff рисунков);
- `tb/<α>/events.*` — TensorBoard, история best-cost по BCO-итерациям
  (у всех четырёх; онлайн: `tensorboard --logdir artifacts/runs/<experiment>`).

Консольные/файловые логи запуска — также в `artifacts/cli_logs/rerun_<exp>.log`.

## Данные

Бенчмарки (Mandl, Mumford0-3) — `datasets/benchmark/`; EKB — `datasets/EKB/`.
Веса моделей — `artifacts/model_weights*/` (грузятся как есть).
