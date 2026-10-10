# Устройство конфигов

Конфиг собирается Hydra из небольших пресетов. Точки входа — файлы в `runs/`.
Они выбирают пресеты и задают только настройки прогона.
Путь группы используется для выбора файла: `training/algorithm=d3po`.
Поля собранного конфига сохраняют свои имена: `algorithm.gamma=0.95`.

```text
cfg/
├── runs/            # search, train, train_construction, train_construction_overnight, generate_construction, generate_edit, init_routes
├── city/instance/   # бенчмарки и ограничения маршрутов
├── network/corruption/ # параметры порчи
├── objectives/      # cost/, adjustment/, preferences/
├── policy/          # model/ с route_generator/; model_profile/
├── training/        # algorithm/, dataset/, dataset_build/, curriculum/
├── search/          # method/, initial/, sweep/
├── common/          # общая симметричность маршрутов
├── run/             # seed запуска
└── env/             # construction / edit; features/ с входами модели
```

- `policy/model/` отвечает, **какая архитектура** строится.
- `policy/model_profile/` отвечает, **какая версия** архитектуры используется, вместе
  с чекпоинтом и пресетом среды для prediction / поиска.
- Правила действий задаются в `env.params`: `force_linking_unlinked` и `max_act_len`
  общие; `max_route_edit_steps`, `max_trim_actions_per_route`, `force_nonhalt_first_step`,
  `allow_trim_below_min` и `forbid_halt_when_overlong` относятся к edit.
  При обучении override имеет вид `env.params.max_route_edit_steps=20`, при поиске —
  `models.edit.env.params.max_route_edit_steps=20`.
- Метод выбирает профили по роли: `construction` или `edit`.
- `method.bees` содержит число пчёл, оператор, профиль и полную маску действий.
  Группы выполняются в порядке записи. Одинаковые операторы не сливаются.
- `method` также задаёт число проходов, шагов мутации и вероятность обрезки.
- `corruption` содержит обязательные параметры порчи; среда и алгоритм их не читают.
- `copytiers` задаёт число событий, виды копий, `max_multiplicity` и ограничение покрытия.
  `realistic` выполняет `dup`, затем `detour`, затем `drop_cover`. Оба пресета
  преобразуются в этапы общего `network.corruption.corrupt`; порядок событий сохраняется.
- `algorithm` задаёт GAE, clip, entropy, diversity и критик. `train_loop` задаёт
  batch, horizon, число итераций, валидацию и бюджет времени. `optimizer` задаёт
  параметры оптимизатора. Обе среды используют один цикл обучения.
- `common.symmetric_routes` задаёт симметричность для State, cost и модели.
  `cost.low_memory_mode` выбирает режим памяти расчёта cost.
- `preferences` задаёт sampler весов: `variable_weights`, `pp_fraction`,
  `op_fraction`, `mcw_fraction`. Порядок целей и фиксированные веса берутся
  из `cost.kwargs` через интерполяции.

Примеры:

```powershell
python scripts/run_search.py --dry-run --methods ea
python scripts/run_search.py --methods nea_edit --instances mandl
python scripts/run.py --config-name runs/train --dry-run
python scripts/run.py --config-name runs/train_construction training/algorithm=d3po --dry-run
python scripts/generate_dataset.py --config-name runs/generate_edit --out datasets/lc_copytiers_manifest_v1 --dry-run
```

Поиск и edit-обучение используют одну цель `objectives/cost=rtt_wmc`. Штраф за изменение
сети задаёт `objectives/adjustment/cap.yaml`. Выключить штраф: `cost.kwargs.penalties.adjustment=0`
(в поиске — `--adj-weight 0`). Режим «цель»: `cost.kwargs.adjustment_degree_objective=target`.
Adjustment рассчитывается по LCS полных последовательностей маршрутов из статьи.
При `cap` штраф равен `weight · max(adjustment − target, 0)`. Поле `cost` в таблицах
и логах включает настроенные штрафы; экспорт использует готовый результат оценки.
Порядок целей задаёт `cost.kwargs.objectives`, дополнительные штрафы —
`cost.kwargs.penalties` после регистрации вычислителя в `objectives/criteria.py`.

`run.type`: `search`, `training`.
Seed запуска задаётся через `run.seed`, выбор CPU — через `run.cpu=true`.
Seeds разбиения, оценки, подготовки модели датасета и порчи задаются отдельно:
`dataset.legacy_split.seed`, `evaluation.seed`, `dataset_build.policy_seed`,
`dataset_build.corruption_seed`.
`features` задаёт имена и порядок входных признаков модели.

Логирование содержит только скаляры: `rollout`, `train`, `eval`, `pareto`, `search`,
`time`. TensorBoard читает общую папку `artifacts/`.

`evaluation.pareto_enabled` включает мониторинг Pareto / HV / EU: включён в
construction, выключен в edit. Частоту задаёт `evaluation.pareto_period`, сетку
мониторинга — `pareto_preference_count`, число стохастических выборок —
`pareto_stochastic_samples`. `hv_reference` задаёт точку отсчёта HV по порядку
активных целей; `eu_preference_count` — размер сетки запросов EU.
`evaluation.final_test` включает итоговую оценку last / best в режимах greedy /
stochastic; её сетку и выборки задают `preference_count` и `stochastic_samples`.
В construction она использует отдельный test и пишет `eval/test/…`; в edit — полную
validation и пишет `eval/final_validation/…`.
`dataset.legacy_split.n_val_per_tier` задаёт число графов каждого тира для текущей
edit-валидации старого датасета. Для нового формата состав выборок хранится в manifest.
`curriculum.enabled` включает расписание `curriculum.schedule`
только для train. Состав validation от этапа curriculum не зависит.
`dataset.train_pipeline` и `dataset.evaluation_pipeline` задают масштабирование,
поворот и отражение графов, а также подготовку признаков construction / edit.
Протокол и выбранные индексы графов записываются в `run.json`, собранные параметры
— в `config.yaml`, значения метрик — в `history.csv` и TensorBoard.

YAML содержит значения и директиву `# @package`, задающую место пресета в собранном конфиге.

Датасеты:

- `dataset.kind` явно выбирает construction / edit; edit требует seed-сети.
- `dataset.source`: `file`, `dynamic` или `benchmark`. Файловые данные задаются
  через `dataset.path`. Новый формат — `dataset.pt` и `manifest.json`; ограничения
  маршрутов и разбиение читаются из manifest. `legacy_limits` задаёт ограничения
  при отсутствии manifest, `legacy_split` — разбиение старых raw / collated / edit-данных.
- Для `dynamic` нужен `dataset.source_params` с параметрами `SyntheticParams`,
  для `benchmark` — `dataset.benchmark_city`. У обоих обязательна отдельная
  `dataset.validation_path`; `dataset.test_path` задаёт отдельный test.
- Генерация запускается `scripts/generate_dataset.py` с `runs/generate_construction`
  или `runs/generate_edit`. Папка обязательна: `--out`. `--dry-run` проверяет входы,
  `--force` пересобирает указанную папку после успешной подготовки нового результата.
  Идентичный полный датасет пропускается; неполные данные и другие параметры
  вызывают ошибку. Seed графов задаёт `run.seed`; размеры и split — `dataset_build`.
- `--convert-from <папка>` читает старый датасет и записывает новый формат в отдельную
  папку без генерации. Путь готового результата передаётся обучению через `dataset.path`.

Профили моделей:

| Профиль | Архитектура | Файл весов относительно `--weights-dir` |
|---|---|---|
| `construction_v1` | `construction` | `inductive_random_graphs_weighted_connectivity.pt` |
| `edit_v1` | `edit_trim_seeded` | `improvement/improvement_lc_rttconn_adj_w10_t02_finetune100.pt` |

Профиль выбирается для роли метода через Hydra override:

```powershell
python scripts/run.py --config-name runs/search search/method=nea_edit "policy/model_profile@models.edit=edit_v1" run.cpu=true --dry-run
```

Профиль подключает архитектуру из `policy/model/` через `defaults`; готовая архитектура
находится в `model_profile.model` или `models.<роль>.model`. Поле `checkpoint_path`
задаёт файл весов. Корень весов передаётся через `--weights-dir`; по умолчанию это
`artifacts/model_weights/`. Для обучения construction доступны архитектуры
`construction` и `construction_holliday`, выбранные в пресетах `runs/`.
