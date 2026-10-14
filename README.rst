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

ConnectPT — исследовательский набор инструментов для проектирования и улучшения
маршрутных сетей общественного транспорта. Основные возможности:

- предобработка данных города: остановки, графы, матрицы времени и спроса;
- оценка сети: ATT, RTT, связность WMC, степень изменения маршрутов;
- поиск маршрутной сети: EA, NEA, NEA-Edit, NEA-Combined;
- обучение нейросетевых политик алгоритмами PPO и D3PO.

Политика ``construction`` строит сеть с нуля. Политика ``edit`` улучшает готовую
сеть: добавляет участки маршрутов, сокращает их начало или конец и завершает
правку. Нейросетевая часть основана на проекте
`transit_learning <https://github.com/AHolliday/transit_learning>`_.

Установка
---------

Требуется Python 3.11 или новее. В активированном виртуальном окружении из корня
репозитория:

.. code-block:: bash

   python -m pip install -r requirements.txt
   python -m pip install -e .

Быстрый старт
-------------

Обучение и поиск запускаются через ``scripts/run.py``. Задачу выбирает конфиг,
параметры меняются аргументами командной строки.

Проверка входных данных и конфига поиска:

.. code-block:: bash

   python scripts/run.py --config-name runs/search --methods ea --instances mandl --dry-run

Поиск на Mandl:

.. code-block:: bash

   python scripts/run.py --config-name runs/search --methods ea --instances mandl --alpha 0.5 --seeds 0

Обучение построению и редактированию сети:

.. code-block:: bash

   python scripts/run.py --config-name runs/train_construction training/algorithm=ppo
   python scripts/run.py --config-name runs/train_edit

Для обучения нужны подготовленные датасеты, для нейросетевого поиска — веса
соответствующих моделей. Подготовка данных, выбор методов и фоновый запуск
описаны в `инструкции скриптов <scripts/README.md>`_. Группы параметров и их
выбор — в `описании конфигов <connectpt/routes_generator/cfg/README.md>`_.

Структура проекта
-----------------

``connectpt/preprocess/``
   Предобработка данных реальных городов.

``connectpt/routes_generator/``
   Генерация и улучшение маршрутных сетей. Модули ``city`` и ``network`` работают
   с городами и маршрутами, ``objectives`` оценивает сеть, ``env`` задаёт среду,
   ``policy`` содержит модели, ``training`` и ``search`` — обучение и поиск,
   ``run`` — сборку и запуск экспериментов, ``common`` — общие функции.

``connectpt/routes_generator/cfg/``
   Конфиги Hydra для задач, моделей, алгоритмов и городов.

``scripts/``
   Запуск экспериментов, подготовка датасетов и начальных сетей.
   ``launch.ps1`` и ``launch.sh`` запускают скрипты в фоне на Windows и Linux.

``datasets/``
   Бенчмарки Mandl и Mumford0–3, локальные обучающие датасеты и начальные сети.

``artifacts/``
   Веса моделей, результаты, логи и TensorBoard.

Пример предобработки
--------------------

.. code-block:: python

   import geopandas as gpd

   from connectpt.preprocess import Modality, preprocess

   blocks = gpd.read_file("path/to/blocks.geojson")
   result, graph = preprocess(blocks, [Modality.BUS])
   stops_gdf, time_matrix, stop_graph = result[Modality.BUS]

Контакты
--------

- Александр Морозов: alexandermorozzov@gmail.com
- Руслан Козляк: rkozliak@gmail.com

.. readme-end

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.11+-blue
   :target: https://www.python.org/
