import numpy as np
from models import Action, ActionType

class Grader:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def grade(self, episode_history: list) -> float:
        try:
            if not isinstance(episode_history, (list, tuple)):
                return 0.5
                
            if self.task_id == "congestion_relief":
                return self._grade_task1(episode_history)
            elif self.task_id == "fair_scheduling":
                return self._grade_task2(episode_history)
            elif self.task_id == "emergency_priority":
                return self._grade_task3(episode_history)
            elif self.task_id == "throughput_maximization":
                return self._grade_task4(episode_history)
            return self._clamp(0.5)
        except Exception:
            # Fallback to a neutral safe score if logic fails
            # Explicitly return as a standard float
            return float(0.5)

    def _unpack_history(self, history: list) -> list:
        """Robustly extracts (obs, act, rew) from history items of any format."""
        unpacked = []
        if not hasattr(history, "__iter__"):
            return unpacked
            
        for item in history:
            try:
                if isinstance(item, (list, tuple)):
                    # Handle (obs, act, rew, ...) - any length >= 3
                    if len(item) >= 3:
                        unpacked.append((item[0], item[1], item[2]))
                elif isinstance(item, dict):
                    # Handle dictionary format
                    obs = item.get("observation") or item.get("obs") or item.get("state")
                    act = item.get("action") or item.get("act")
                    rew = item.get("reward") or item.get("rew")
                    if obs is not None:
                        unpacked.append((obs, act, rew))
            except Exception:
                continue
        return unpacked

    def _get_val(self, obj, key, default=0):
        """Robustly gets a value from an object (attribute) or dict (key)."""
        if obj is None:
            return default
        # If it's a dict, use .get()
        if isinstance(obj, dict):
            return obj.get(key, default)
        # If it has the attribute, use getattr()
        if hasattr(obj, key):
            val = getattr(obj, key)
            return val() if callable(val) else val
        # Try as a dict if it hasn't worked yet (handles some Pydantic-to-dict cases)
        try:
            return obj[key]
        except (KeyError, TypeError):
            pass
        return default

    def _clamp(self, score: float) -> float:
        """Clamps score to (0.1, 0.9) interval and handles NaN/None/Inf/Non-numeric."""
        try:
            # Handle list/array inputs by taking mean if necessary
            if isinstance(score, (list, np.ndarray)):
                if len(score) == 0:
                    return 0.5
                score = np.mean(score)
            
            # Robust numeric check
            if score is None or not isinstance(score, (int, float, np.number)):
                return 0.5
                
            # Convert to float early to avoid numpy scalar issues
            val = float(score)
            
            if np.isnan(val) or np.isinf(val):
                return 0.5
                
            # Strictly between 0 and 1. We use a safe buffer [0.15, 0.85].
            # This ensures we are never 0.0 or 1.0 even with rounding.
            # Convert to standard Python float for validator compatibility.
            clamped = float(np.clip(val, 0.15, 0.85))
            
            # Final sanity check against NaN after clip (rare but defensive)
            if np.isnan(clamped):
                return 0.5
            return clamped
        except Exception:
            return 0.5

    def _grade_task1(self, history: list) -> float:
        # Objective: Reduce total queue length.
        clean_history = self._unpack_history(history)
        if not clean_history:
            return self._clamp(0.5)
            
        queues = []
        for obs, act, rew in clean_history:
            total_q = (
                self._get_val(obs, "north_queue", 0) + 
                self._get_val(obs, "south_queue", 0) + 
                self._get_val(obs, "east_queue", 0) + 
                self._get_val(obs, "west_queue", 0)
            )
            queues.append(total_q)
            
        if not queues:
            return self._clamp(0.5)
            
        avg_queue = float(np.mean(queues))
        # if avg_queue is 0, score 0.85. If avg_queue > 50, score 0.15.
        score = 1.0 - (avg_queue / 50.0)
        return self._clamp(score)

    def _grade_task2(self, history: list) -> float:
        # Objective: Balance traffic flow / Fairness.
        clean_history = self._unpack_history(history)
        if not clean_history:
            return self._clamp(0.5)
            
        lane_waits = []
        for obs, act, rew in clean_history:
            waits = [
                self._get_val(obs, "north_wait", 0),
                self._get_val(obs, "south_wait", 0),
                self._get_val(obs, "east_wait", 0),
                self._get_val(obs, "west_wait", 0)
            ]
            lane_waits.append(np.std(waits))
        
        if not lane_waits:
            return self._clamp(0.5)
            
        avg_std = float(np.mean(lane_waits))
        # Low std -> High score. Cap denominator at 100.
        score = 1.0 - (avg_std / 100.0)
        return self._clamp(score)

    def _grade_task3(self, history: list) -> float:
        # Objective: Emergency veh prioritization.
        clean_history = self._unpack_history(history)
        if not clean_history:
            return self._clamp(0.5)

        emergency_durations = []
        current_emerg = False
        start_step = 0
        
        for i, (obs, act, rew) in enumerate(clean_history):
            raw_active = self._get_val(obs, "emergency_active", False)
            # Handle NaN being truthy
            is_active = bool(raw_active)
            if isinstance(raw_active, float) and np.isnan(raw_active):
                is_active = False
                
            if is_active and not current_emerg:
                current_emerg = True
                start_step = i
            elif not is_active and current_emerg:
                current_emerg = False
                emergency_durations.append(i - start_step)
        
        if not emergency_durations:
            # If no emergency ended, check if any started and never finished
            if current_emerg:
                return self._clamp(0.25) # Penalize for never finishing
            return self._clamp(0.75) # Safe high score
            
        avg_duration = float(np.mean(emergency_durations))
        if avg_duration <= 0: return self._clamp(0.85)
        # Fast clearing (< 10 steps) -> High score
        score = 1.0 - (avg_duration / 30.0)
        return self._clamp(score)

    def _grade_task4(self, history: list) -> float:
        # Objective: Throughput Maximization.
        clean_history = self._unpack_history(history)
        if not clean_history:
            return self._clamp(0.5)
            
        final_served = 0.0
        for obs, act, rew in clean_history:
            raw_served = self._get_val(obs, "vehicles_served", 0)
            # Handle NaN in max
            if isinstance(raw_served, (int, float, np.number)) and not np.isnan(raw_served):
                final_served = max(final_served, float(raw_served))
            
        steps = len(clean_history)
        if steps == 0: return self._clamp(0.5)
        
        ratio = final_served / float(steps)
        # 0.5 vehicles/step is a decent baseline for 0.5 score
        score = ratio / 1.0 
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

def grade_throughput_maximization(history):
    return run_grading(history, "throughput_maximization")

if __name__ == "__main__":
    from env import TrafficEnvironment # Local import to avoid circular top-level
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
    print(f"Task 4 Score: {grade_throughput_maximization(history)}")
    
    # Empty history test
    print(f"Empty Score: {grade_congestion_relief([])}")
