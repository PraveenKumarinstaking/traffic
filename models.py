from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class TrafficPhase(str, Enum):
    NS_GREEN = "NS_GREEN"  # North-South Green
    EW_GREEN = "EW_GREEN"  # East-West Green
    NS_YELLOW = "NS_YELLOW"
    EW_YELLOW = "EW_YELLOW"

class ActionType(str, Enum):
    KEEP_PHASE = "KEEP_PHASE"
    SWITCH_NS_GREEN = "SWITCH_NS_GREEN"
    SWITCH_EW_GREEN = "SWITCH_EW_GREEN"
    EXTEND_GREEN = "EXTEND_GREEN"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"

class Action(BaseModel):
    action: ActionType

class Observation(BaseModel):
    north_queue: int
    south_queue: int
    east_queue: int
    west_queue: int
    north_wait: float
    south_wait: float
    east_wait: float
    west_wait: float
    current_phase: str
    phase_duration: int
    emergency_active: bool
    emergency_lane: Optional[str] # "north", "south", "east", "west" or None
    total_wait_time: float
    vehicles_served: int
    total_congestion_score: float

class Reward(BaseModel):
    value: float
    components: dict # For debugging reward shaping
