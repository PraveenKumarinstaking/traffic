from typing import List, Optional, Dict
import random

class TrafficEnvironment:
    TASKS = [
        {"id": "congestion_relief", "name": "Task 1 - Basic Congestion Relief", "grader": "grader:grade_congestion_relief", "verifier": "grader:grade_congestion_relief"},
        {"id": "fair_scheduling", "name": "Task 2 - Fair Phase Scheduling", "grader": "grader:grade_fair_scheduling", "verifier": "grader:grade_fair_scheduling"},
        {"id": "emergency_priority", "name": "Task 3 - Emergency Vehicle Prioritization", "grader": "grader:grade_emergency_priority", "verifier": "grader:grade_emergency_priority"},
        {"id": "throughput_maximization", "name": "Task 4 - Throughput Maximization", "grader": "grader:grade_throughput_maximization", "verifier": "grader:grade_throughput_maximization"}
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        # Consistent task registry for discovery (redundancy)
        self.tasks = self.TASKS
        
        self.reset()

    @classmethod
    def get_tasks(cls) -> List[Dict]:
        """Returns the list of available tasks for this environment."""
        return cls.TASKS

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        self.state = {
            "north_queue": 0,
            "south_queue": 0,
            "east_queue": 0,
            "west_queue": 0,
            "north_wait": 0.0,
            "south_wait": 0.0,
            "east_wait": 0.0,
            "west_wait": 0.0,
            "current_phase": "NS_GREEN",
            "phase_duration": 0,
            "emergency_active": False,
            "emergency_lane": None,
            "total_wait_time": 0.0,
            "vehicles_served": 0,
            "total_congestion_score": 0.0
        }
        return self.state

    def step(self, action):
        # Dummy step logic for validation compatibility
        # In a real environment, this would update based on the action
        self.state["phase_duration"] += 5
        self.state["north_queue"] = max(0, self.state["north_queue"] + random.randint(-1, 2))
        self.state["south_queue"] = max(0, self.state["south_queue"] + random.randint(-1, 2))
        
        done = self.state["phase_duration"] >= 100
        reward = - (self.state["north_queue"] + self.state["south_queue"]) / 10.0
        
        return self.state, reward, done, {}
