"""Constants owned by this package."""

GOAL_NAMES = ("ATT", "RTT", "WMC")

COST_WEIGHT_KEY_ORDER = (
    "demand_time_weight",
    "route_time_weight",
    "median_connectivity_weight",
)

COST_COMPONENT_NAMES = ("demand", "route", "connectivity")

METRIC_NAMES = (
    "cost",
    "ATT",
    "RTT",
    "d0",
    "d1",
    "d2",
    "d_un",
    "n_disconnected",
    "stops_oob",
    "WMC",
)

METRIC_TAG_ALIASES = {"ATT": "att_minutes", "RTT": "rtt_minutes"}

STATE_PENALTIES = ("constraints", "adjustment")

ACTION_PENALTIES = ("edit_step", "forced_halt")

PENALTIES = STATE_PENALTIES + ACTION_PENALTIES

PREFERENCE_KEYS = {"ATT": "demand_time_weight", "RTT": "route_time_weight", "WMC": "median_connectivity_weight"}
