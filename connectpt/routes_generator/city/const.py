"""Constants owned by this package."""

STOP_KEY = "stop"

STREET_KEY = (STOP_KEY, "street", STOP_KEY)

DEMAND_KEY = (STOP_KEY, "demand", STOP_KEY)

ROUTE_KEY = (STOP_KEY, "route", STOP_KEY)

DMD_FEAT_IDX = 0

SHORTESTPATH_FEAT_IDX = 1

OUT_DEMAND_FEAT_IDX = 4

IN_DEMAND_FEAT_IDX = 5

SPEED_MPS = 15.0

RAW_GRAPH_FILENAME = "raw_graphs_1000.pkl"

MST = "mst"

OUT_KNN = "outgoing_knn"

IN_KNN = "incoming_knn"

VORONOI = "voronoi"

GRID4 = "4_grid"

GRID8 = "8_grid"

CIRC = "circulant"

SMALLWORLD = "small_world"

MIXED = "mixed"

MAX_DEMAND = 800.0

MIN_DEMAND = 60.0

MAX_POP_STD_M = 15_000

MIN_POP_STD_M = 2000

MAX_SINK_RADIUS_M = 7_500

MIN_SINK_RADIUS_M = 1_500

CENTERNODES_PER_NODE = 1 / 10

SIDE_LENGTH_M = 30_000

EKB_COORD_CRS = "EPSG:32641"

TRAVEL_TIME_SCALE = 60.0  # files are in minutes, the cost modules work in seconds
