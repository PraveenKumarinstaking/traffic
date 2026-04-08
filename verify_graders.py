from grader import grade_congestion_relief, grade_fair_scheduling, grade_emergency_priority, grade_throughput_maximization

def test_grader(name, grader_fn):
    print(f"Testing {name}...")
    
    # 1. Empty history
    score_empty = grader_fn([])
    print(f"  Empty history: {score_empty}")
    assert 0.2 <= score_empty <= 0.8
    
    # 2. Normal history (List of dicts)
    mock_hist = [
        {"observation": {"north_queue": 5, "south_queue": 5, "east_queue": 5, "west_queue": 5, 
                         "north_wait": 10.0, "south_wait": 10.0, "east_wait": 10.0, "west_wait": 10.0,
                         "emergency_active": False, "vehicles_served": 10}, 
         "action": None, "reward": 0.0}
    ]
    score_normal = grader_fn(mock_hist)
    print(f"  Normal history: {score_normal}")
    assert 0.2 <= score_normal <= 0.8
    
    # 3. Extreme low
    mock_hist_bad = [
        {"observation": {"north_queue": 1000, "south_queue": 1000, "east_queue": 1000, "west_queue": 1000, 
                         "north_wait": 1000.0, "south_wait": 1000.0, "east_wait": 1000.0, "west_wait": 1000.0,
                         "emergency_active": True, "vehicles_served": 0}, 
         "action": None, "reward": -100.0}
    ]
    score_bad = grader_fn(mock_hist_bad)
    print(f"  Bad history: {score_bad}")
    assert 0.2 <= score_bad <= 0.8
    
    # 4. NaN input (Dict with NaN)
    mock_hist_nan = [
        {"observation": {"north_queue": float('nan'), "emergency_active": float('nan')}, 
         "action": None, "reward": 0.0}
    ]
    score_nan = grader_fn(mock_hist_nan)
    print(f"  NaN history: {score_nan}")
    assert 0.2 <= score_nan <= 0.8

if __name__ == "__main__":
    test_grader("Congestion Relief", grade_congestion_relief)
    test_grader("Fair Scheduling", grade_fair_scheduling)
    test_grader("Emergency Priority", grade_emergency_priority)
    test_grader("Throughput Maximization", grade_throughput_maximization)
    
    print("\nALL ROBUSTNESS TESTS PASSED!")
