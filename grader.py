import math

class Grader:
    def __init__(self, task_id):
        self.task_id = task_id

    def grade(self, episode_history):
        """Entry point for grading. Wraps logic in top-level try-except."""
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
            return self._sanitize(0.5)
        except Exception:
            return 0.5

    def _unpack_history(self, history):
        """Robustly extracts (obs, act, rew) using duck-typing to avoid imports."""
        unpacked = []
        if not hasattr(history, "__iter__"):
            return unpacked
            
        for item in history:
            try:
                # Handle tuple/list format: (obs, act, rew, ...)
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    unpacked.append((item[0], item[1], item[2]))
                # Handle dictionary format
                elif isinstance(item, dict):
                    obs = item.get("observation") or item.get("obs") or item.get("state")
                    act = item.get("action") or item.get("act")
                    rew = item.get("reward") or item.get("rew")
                    if obs is not None:
                        unpacked.append((obs, act, rew))
                # Handle attribute access (e.g. if item is an object)
                elif hasattr(item, "observation") or hasattr(item, "obs"):
                    obs = getattr(item, "observation", None) or getattr(item, "obs", None)
                    act = getattr(item, "action", None) or getattr(item, "act", None)
                    rew = getattr(item, "reward", None) or getattr(item, "rew", None)
                    if obs is not None:
                        unpacked.append((obs, act, rew))
            except Exception:
                continue
        return unpacked

    def _get_val(self, obj, key, default=0):
        """Gets a numeric value from a variety of object types safely."""
        if obj is None:
            return default
        try:
            # Try dict-style access
            if isinstance(obj, dict):
                val = obj.get(key, default)
            # Try attribute-style access for Pydantic models or objects
            elif hasattr(obj, key):
                val = getattr(obj, key)
                if callable(val): val = val()
            # Try index-style if it's a list (unlikely for keys but defensive)
            else:
                try:
                    val = obj[key]
                except (KeyError, TypeError, IndexError):
                    val = default
            
            # Sanitization of the value itself (ensure it's a number)
            if val is None: return default
            if isinstance(val, bool): return int(val)
            if isinstance(val, (int, float)):
                if val != val: return default # NaN check
                return val
            return default
        except Exception:
            return default

    def _sanitize(self, val):
        """Final output sanitization using pure Python. Target range [0.2, 0.8]."""
        try:
            if val is None: return 0.5
            f_val = float(val)
            # Handle NaN/Inf
            if f_val != f_val or f_val == float('inf') or f_val == float('-inf'):
                return 0.5
            # Clamp to [0.2, 0.8]
            clamped = max(0.2, min(0.8, f_val))
            # Round to 3 decimal places for clean float representation
            return round(clamped, 3)
        except (TypeError, ValueError):
            return 0.5

    def _grade_task1(self, history):
        # Objective: Congestion Relief (Queue length)
        clean = self._unpack_history(history)
        if not clean: return self._sanitize(0.5)
        
        queues = []
        for obs, act, rew in clean:
            q = (self._get_val(obs, "north_queue") + self._get_val(obs, "south_queue") + 
                 self._get_val(obs, "east_queue") + self._get_val(obs, "west_queue"))
            queues.append(q)
            
        if not queues: return self._sanitize(0.5)
        avg_q = sum(queues) / len(queues)
        # Score: 0.8 if avg_q=0, 0.2 if avg_q >= 100
        score = 1.0 - (avg_q / 100.0)
        return self._sanitize(score)

    def _grade_task2(self, history):
        # Objective: Fairness (Wait time balance)
        clean = self._unpack_history(history)
        if not clean: return self._sanitize(0.5)
        
        stds = []
        for obs, act, rew in clean:
            waits = [
                self._get_val(obs, "north_wait"), self._get_val(obs, "south_wait"),
                self._get_val(obs, "east_wait"), self._get_val(obs, "west_wait")
            ]
            mean_w = sum(waits) / len(waits)
            var_w = sum((x - mean_w)**2 for x in waits) / len(waits)
            stds.append(math.sqrt(var_w))
            
        if not stds: return self._sanitize(0.5)
        avg_std = sum(stds) / len(stds)
        # Score: 1.0 if std=0, 0.0 if std >= 100. Range handled by sanitize.
        score = 1.0 - (avg_std / 100.0)
        return self._sanitize(score)

    def _grade_task3(self, history):
        # Objective: Emergency Priority
        clean = self._unpack_history(history)
        if not clean: return self._sanitize(0.5)
        
        current_emerg = False
        duration = 0
        cleared_durations = []
        
        for i, (obs, act, rew) in enumerate(clean):
            is_active = bool(self._get_val(obs, "emergency_active", False))
            if is_active:
                if not current_emerg:
                    current_emerg = True
                    duration = 0
                duration += 1
            else:
                if current_emerg:
                    current_emerg = False
                    cleared_durations.append(duration)
        
        if not cleared_durations:
            if current_emerg: return self._sanitize(0.3) # Active but never cleared
            return self._sanitize(0.7) # No emergency occurred (safe high passing)
            
        avg_dur = sum(cleared_durations) / len(cleared_durations)
        score = 1.0 - (avg_dur / 50.0)
        return self._sanitize(score)

    def _grade_task4(self, history):
        # Objective: Throughput
        clean = self._unpack_history(history)
        if not clean: return self._sanitize(0.5)
        
        max_served = 0
        for obs, act, rew in clean:
            served = self._get_val(obs, "vehicles_served")
            if served > max_served: max_served = served
            
        ratio = max_served / max(1, len(clean))
        score = ratio / 1.0 # 1 vehicle per step is a very high throughput
        return self._sanitize(score)

def run_grading(history, task_id):
    return Grader(task_id).grade(history)

# Entry points
def grade_congestion_relief(history): return run_grading(history, "congestion_relief")
def grade_fair_scheduling(history): return run_grading(history, "fair_scheduling")
def grade_emergency_priority(history): return run_grading(history, "emergency_priority")
def grade_throughput_maximization(history): return run_grading(history, "throughput_maximization")

if __name__ == "__main__":
    # Internal smoke test with dummy history
    h = [{"obs": {"north_queue": 10}, "act": None, "rew": 0} for _ in range(10)]
    print(f"Task 1: {grade_congestion_relief(h)}")
    print(f"Task 2: {grade_fair_scheduling(h)}")
    print(f"Task 3: {grade_emergency_priority(h)}")
    print(f"Task 4: {grade_throughput_maximization(h)}")
    print(f"Empty: {grade_congestion_relief([])}")
