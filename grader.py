import numpy as np
from env import TrafficEnvironment
from models import Action, ActionType

class Grader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, episode_history: list) -> float:
        if self.task_id == "congestion_relief":
            return self._grade_task1(episode_history)
        elif self.task_id == "fair_scheduling":
            return self._grade_task2(episode_history)
        elif self.task_id == "emergency_priority":
            return self._grade_task3(episode_history)
        return self._clamp(0.0)

    def _clamp(self, score: float) -> float:
        """Clamps score to (0.01, 0.99) interval and handles NaN."""
        if np.isnan(score):
            return 0.5
        return float(np.clip(score, 0.01, 0.99))

    def _grade_task1(self, history: list) -> float:
        # Objective: Reduce total queue length.
        if not history:
            return self._clamp(0.5)
            
        # compare avg queue length to a baseline (~10-15 vehicles)
        queues = [sum([obs.north_queue, obs.south_queue, obs.east_queue, obs.west_queue]) for obs, act, rew in history]
        avg_queue = np.mean(queues)
        
        # if avg_queue is 0, score 0.99. If avg_queue > 25, score 0.01.
        score = 1.0 - (avg_queue / 25.0)
        return self._clamp(score)

    def _grade_task2(self, history: list) -> float:
        # Objective: Balance traffic flow / Fairness.
        if not history:
            return self._clamp(0.5)
            
        # Measure variance in queue lengths or wait times.
        lane_waits = []
        for obs, act, rew in history:
            waits = [obs.north_wait, obs.south_wait, obs.east_wait, obs.west_wait]
            lane_waits.append(np.std(waits))
        
        if not lane_waits:
            return self._clamp(0.5)
            
        avg_std = np.mean(lane_waits)
        # Low std -> High score
        score = 1.0 - (avg_std / 50.0)
        return self._clamp(score)

    def _grade_task3(self, history: list) -> float:
        # Objective: Emergency veh prioritization.
        # Track steps where emergency was active and if it was served.
        emergencies = []
        emergency_durations = []
        current_emerg = False
        start_step = 0
        
        for i, (obs, act, rew) in enumerate(history):
            if obs.emergency_active and not current_emerg:
                current_emerg = True
                start_step = i
            elif not obs.emergency_active and current_emerg:
                current_emerg = False
                emergency_durations.append(i - start_step)
        
        if not emergency_durations:
            # High score for avoiding unnecessary overrides if no emergency
            return self._clamp(0.95)
            
        avg_duration = np.mean(emergency_durations)
        # Fast clearing (< 10 steps) -> High score
        score = 1.0 - (avg_duration / 30.0)
        return self._clamp(score)

def run_grading(history, task_id):
    g = Grader(task_id)
    return g.grade(history)

# Task-specific entry points for compatibility
def grade_congestion_relief(history):
    return run_grading(history, "congestion_relief")

def grade_fair_scheduling(history):
    return run_grading(history, "fair_scheduling")

def grade_emergency_priority(history):
    return run_grading(history, "emergency_priority")

if __name__ == "__main__":
    # Smoke test
    env = TrafficEnvironment(seed=42)
    obs = env.reset()
    history = []
    for _ in range(50):
        # random policy
        action = Action(action=ActionType.KEEP_PHASE)
        next_obs, reward, done, info = env.step(action)
        history.append((obs, action, reward))
        obs = next_obs
        if done: break
        
    print(f"Task 1 Score: {grade_congestion_relief(history)}")
    print(f"Task 2 Score: {grade_fair_scheduling(history)}")
    print(f"Task 3 Score: {grade_emergency_priority(history)}")
    
    # Empty history test
    print(f"Empty Score: {grade_congestion_relief([])}")
