ConnectPT
=========

.. logo-start

.. figure:: https://psv4.userapi.com/s/v1/d2/aE8kEC2MYzxzxQgGbG4SIXKijfv-ouCe9jNMag7ONZ8TdctZo5IKBe-MR2OTRxVEWMIaq7yxqnPSpyKEK4HMDw_yf5_XLgYa-7MQxgABQBIUzMCtFT7G5FsrWZN7GbfnTQUP-1X-NSqK/connectpt_logo_gen_gpt_v4.png
   :alt: ConnectPT

.. logo-end

|PythonVersion|

.. readme-start

Обзор
-----

ConnectPT — исследовательский набор инструментов для проектирования маршрутных
сетей общественного транспорта. В нём есть:

- предобработка реальных городов: остановки, графы, OD-матрицы;
- стоимость сети и метрики: ATT, RTT, связность WMC, степень изменения сети;
- поиск по методу пчелиной колонии: EA, NEA, NEA-Edit, NEA-Combined;
- нейросетевые генераторы маршрутов на основе проекта ``transit_learning``.

Нейросетевая часть обучается в двух режимах:

``construction``
   Модель строит сеть с нуля: начинает с пустой сети и планирует маршруты по
   одному. Алгоритмы: PPO, decomposed PPO, D3PO.

``edit``
   Модель получает готовую сеть и правит каждый маршрут типизированными
   действиями: extend, trim start, trim end, halt. Алгоритмы: PPO, D3PO.


Установка
---------

Установка из GitHub:

.. code-block:: bash

   pip install git+https://github.com/alexandermorozzov/connectpt@main

Для разработки создайте виртуальное окружение и установите пакет в режиме
редактирования:

.. code-block:: bash

   pip install -e .


Быстрый старт
-------------

Все сценарии запускаются скриптами из ``scripts/``. Подробно — в
``scripts/README.md``.

.. code-block:: bash

   python scripts/run_search.py --dry-run --methods ea nea nea_edit nea_combined
   python scripts/run_search.py --methods nea_edit --instances mandl
   python scripts/prepare_data.py
   python scripts/run_training.py
   python scripts/run_training.py --config-name runs/train_construction trainer=d3po_construction


Структура проекта
-----------------

``connectpt/preprocess/``
   Предобработка реальных городов.

``connectpt/routes_generator/``
   Генерация маршрутов. Подпакеты: ``core`` (пути, запуск, чекпоинты),
   ``search`` (пчелиная колония), ``training`` (обучение), ``evaluation``
   (оценка), ``objectives`` (стоимость), ``data`` (инстансы и маршруты),
   ``reports`` (таблицы и рисунки).

``connectpt/routes_generator/models.py``
   Нейросетевые генераторы. ``PathCombiningRouteGenerator`` строит сеть,
   ``TrimPathCombiningRouteGenerator`` добавляет действия правки.

``connectpt/routes_generator/transit_time_estimator.py``
   ``RouteGenBatchState`` (состояние сети) и модули стоимости.

``connectpt/routes_generator/cfg/``
   Конфиги Hydra. Устройство — ``cfg/README.md``.

``datasets/``
   Бенчмарки, начальные сети, обучающие датасеты.

``artifacts/``
   Результаты, веса, логи, TensorBoard.


Пример предобработки
--------------------

.. code-block:: python

   import geopandas as gpd

   from connectpt.preprocess import Modality, preprocess

   blocks = gpd.read_file("path/to/blocks.geojson")
   result, graph = preprocess(blocks, [Modality.BUS])
   stops_gdf, time_matrix, stop_graph = result[Modality.BUS]


Edit-политика
-------------

Construction-модель выбирает пару узлов ``[u, v]``: состояние добавляет к
текущему маршруту кратчайший путь между ними. Пара ``[-1, -1]`` (halt)
завершает маршрут. Такая модель умеет только добавлять участки. Edit-политика
также умеет удалять участки готового маршрута.

Как обрабатывается готовая сеть
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Начальные маршруты — тензор ``batch_size × n_routes × max_route_len``,
заполнитель ``-1``. Для каждого маршрута ``i``:

1. Создаётся новое ``RouteGenBatchState``.
2. Все маршруты, кроме ``i``, добавляются как готовые.
3. Маршрут ``i`` становится текущим (``set_current_routes``).
4. Модель выбирает действия для текущего маршрута.
5. Результат сохраняется в слот ``i``.

После обхода всех слотов маршруты собираются в исходном порядке, и метрики
считаются на полной сети. Модель видит всю сеть при правке каждого маршрута.

``target_n_routes`` добавляет пустые слоты. Пустые слоты штрафуются, только
если ``ignore_stops_oob = False``: штраф ``stops_oob`` учитывает и
незапущенные маршруты, и маршруты короче ``min_route_len``.

Типизированные действия
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ROUTE_ACTION_EXTEND = 0
   ROUTE_ACTION_TRIM_START = 1
   ROUTE_ACTION_TRIM_END = 2
   ROUTE_ACTION_HALT = 3

Шаг задают два тензора: ``action_kinds`` (форма ``batch_size``) и
``path_indices`` (форма ``batch_size × 2``).

``EXTEND``
   ``[u, v]`` — кратчайший путь от ``u`` до ``v`` пристраивается к началу или
   концу маршрута. У пустого маршрута он становится первым участком.

``TRIM_START``
   Удаляет начало маршрута. Новое начало — ``path_indices[:, 1]``.

``TRIM_END``
   Удаляет конец маршрута. Новый конец — ``path_indices[:, 0]``.

``HALT``
   Завершает маршрут. ``path_indices = [-1, -1]``.

Trim не делает маршрут короче ``min_route_len``. Extend не делает его длиннее
``max_route_len``.

Кодирование выбора:

.. code-block:: text

   extend(u, v)      = u * n_nodes + v
   trim_start(u, v)  = 1 * n_nodes * n_nodes + u * n_nodes + v
   trim_end(u, v)    = 2 * n_nodes * n_nodes + u * n_nodes + v

При ``serial_halting = False`` halt — ещё один кандидат того же распределения.
При ``serial_halting = True`` сначала выбирается «продолжить или остановиться».

Halt разрешён, когда длина маршрута не меньше ``min_route_len``. Halt
обязателен, если допустимых правок нет. ``max_route_edit_steps`` (по умолчанию
``2 * max_route_len``) ограничивает число правок одного маршрута: так жадная
оценка не зацикливается на trim / extend.

Оценка edit-политики
~~~~~~~~~~~~~~~~~~~~

``evaluate_lc_improvement`` возвращает:

``seed_cost`` / ``final_cost``
   Средняя стоимость до и после правки.

``delta``
   ``seed_cost - final_cost``. Больше нуля — сеть стала дешевле.

``win_rate``
   Доля графов, где стоимость снизилась.

``changed_route_rate`` / ``changed_graph_rate``
   Доля изменённых маршрутов / графов. Если ``changed_route_rate`` всегда 0,
   политика сразу останавливается.


Контакты
--------

- Александр Морозов: alexandermorozzov@gmail.com
- Руслан Козляк: rkozliak@gmail.com


Благодарности
-------------

Работа выполнена при поддержке Министерства экономического развития
Российской Федерации (ИГК 000000C313925P4C0002), соглашение № 139-15-2025-010.

.. readme-end

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.11+-blue
   :target: https://www.python.org/
