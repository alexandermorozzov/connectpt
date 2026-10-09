from enum import Enum
from typing import Dict, Any

class Modality(Enum):
    """Enumeration of supported public transport modalities."""
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"

# Mapping of modality → OpenStreetMap tags for stop extraction
MODALITY_STOP_TAGS: Dict[Modality, Dict[str, Any]] = {
    Modality.BUS: {"highway": "bus_stop"},
    Modality.TRAM: {"railway": "tram_stop"},
    Modality.TROLLEYBUS: {"highway": "bus_stop", "trolleybus": "yes"},
}

# Mapping of modality → OpenStreetMap filters for line extraction
MODALITY_LINE_TAGS: Dict[Modality, str | None] = {
    Modality.BUS: None,                        # No specific line tag for bus (use driving roads)
    Modality.TRAM: '["railway"="tram"]',
    Modality.TROLLEYBUS: '["trolley_wire"="yes"]'
}
