import numpy as np
from grader import grade_congestion_relief, grade_fair_scheduling, grade_emergency_priority, grade_throughput_maximization
from models import Observation, Action, ActionType

def create_mock_obs(q=0, w=0, ea=False):
    return Observation(
        north_queue=q, south_queue=q, east_queue=q, west_queue=q,
        north_wait=w, south_wait=w, east_wait=w, west_wait=w,
        current_phase="NS_GREEN", phase_duration=0,
        emergency_active=ea, emergency_lane=None,
        total_wait_time=0.0, vehicles_served=0, total_congestion_score=0.0
    )

def test_grader(name, grader_fn):
    print(f"\nTesting {name}...")
    
    # 1. Empty history
    score_empty = grader_fn([])
    print(f"  Empty history: {score_empty}")
    assert 0.0 < score_empty < 1.0, f"{name} failed empty history"
    assert score_empty != 0.0 and score_empty != 1.0
    
    # 2. Normal history
    mock_hist = [(create_mock_obs(q=2), Action(action=ActionType.KEEP_PHASE), 0.1) for _ in range(10)]
    score_normal = grader_fn(mock_hist)
    print(f"  Normal history: {score_normal}")
    assert 0.0 < score_normal < 1.0
    
    # 3. Extreme low (High congestion)
    mock_hist_bad = [(create_mock_obs(q=100), Action(action=ActionType.KEEP_PHASE), -10.0) for _ in range(10)]
    score_bad = grader_fn(mock_hist_bad)
    print(f"  Bad history: {score_bad}")
    assert 0.0 < score_bad < 1.0
    assert score_bad >= 0.1 # Should be clamped to 0.1
    
    # 4. Extreme high (Zero congestion)
    mock_hist_perfect = [(create_mock_obs(q=0), Action(action=ActionType.KEEP_PHASE), 1.0) for _ in range(10)]
    score_perfect = grader_fn(mock_hist_perfect)
    print(f"  Perfect history: {score_perfect}")
    assert 0.0 < score_perfect < 1.0
    assert 0.1 <= score_perfect <= 0.9 # Should be clamped to 0.9
    
    # 5. Dictionary-based history (for JSON telemetry compatibility)
    dict_hist = [
        ({"north_queue": 5, "south_queue": 5, "east_queue": 5, "west_queue": 5, "emergency_active": False}, Action(action=ActionType.KEEP_PHASE), 0.0)
    ]
    score_dict = grader_fn(dict_hist)
    print(f"  Dictionary history: {score_dict}")
    assert 0.1 <= score_dict <= 0.9

    # 6. 4-tuple history (obs, act, rew, info)
    tuple4_hist = [
        (create_mock_obs(q=2), Action(action=ActionType.KEEP_PHASE), 0.1, {"info": "extra"})
    ]
    score_tuple4 = grader_fn(tuple4_hist)
    print(f"  4-tuple history: {score_tuple4}")
    assert 0.1 <= score_tuple4 <= 0.9

if __name__ == "__main__":
    test_grader("Congestion Relief", grade_congestion_relief)
    test_grader("Fair Scheduling", grade_fair_scheduling)
    test_grader("Emergency Priority", grade_emergency_priority)
    test_grader("Throughput Maximization", grade_throughput_maximization)
    
    # Special Task 3 case: Emergency active but never cleared
    print("\nTesting Emergency Priority (Active but not cleared)...")
    hist_active = [(create_mock_obs(ea=True), Action(action=ActionType.KEEP_PHASE), 0.0) for _ in range(10)]
    score_active = grade_emergency_priority(hist_active)
    print(f"  Active but not cleared: {score_active}")
    assert 0.0 < score_active < 1.0
    
    print("\nALL TESTS PASSED!")
