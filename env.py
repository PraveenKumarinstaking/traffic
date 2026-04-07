import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from models import Observation, Action, ActionType, TrafficPhase, Reward

class TrafficEnvironment:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.max_steps = 100
        self.yellow_time = 2  # steps
        self.min_green_time = 5 # steps
        
        # Consistent task registry for discovery
        self.tasks = [
            {"id": "congestion_relief", "name": "Task 1 — Basic Congestion Relief", "grader": "grader:grade_congestion_relief"},
            {"id": "fair_scheduling", "name": "Task 2 — Fair Phase Scheduling", "grader": "grader:grade_fair_scheduling"},
            {"id": "emergency_priority", "name": "Task 3 — Emergency Vehicle Prioritization", "grader": "grader:grade_emergency_priority"},
            {"id": "throughput_maximization", "name": "Task 4 — Throughput Maximization", "grader": "grader:grade_throughput_maximization"}
        ]
        
        self.reset()

    @classmethod
    def get_tasks(cls) -> List[Dict]:
        """Returns the list of available tasks for this environment."""
        return [
            {"id": "congestion_relief", "name": "Task 1 — Basic Congestion Relief", "grader": "grader:grade_congestion_relief"},
            {"id": "fair_scheduling", "name": "Task 2 — Fair Phase Scheduling", "grader": "grader:grade_fair_scheduling"},
            {"id": "emergency_priority", "name": "Task 3 — Emergency Vehicle Prioritization", "grader": "grader:grade_emergency_priority"},
            {"id": "throughput_maximization", "name": "Task 4 — Throughput Maximization", "grader": "grader:grade_throughput_maximization"}
        ]

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.steps = 0
        self.lanes = {
            "north": {"queue": 0, "wait_times": []},
            "south": {"queue": 0, "wait_times": []},
            "east": {"queue": 0, "wait_times": []},
            "west": {"queue": 0, "wait_times": []}
        }
        
        self.current_phase = TrafficPhase.NS_GREEN
        self.phase_duration = 0
        self.in_transition = False
        self.next_phase = None
        self.transition_timer = 0
        
        self.emergency_vehicle = None # {"lane": "north", "time_arrived": 0}
        self.vehicles_served = 0
        self.total_wait_time = 0.0
        
        # Traffic arrival rates (vehicles per step)
        self.arrival_rates = {
            "north": 0.3,
            "south": 0.3,
            "east": 0.2,
            "west": 0.2
        }
        
        return self._get_observation()

    def _get_observation(self) -> Observation:
        total_wait = sum(sum(l["wait_times"]) for l in self.lanes.values())
        total_queue = sum(l["queue"] for l in self.lanes.values())
        
        obs = Observation(
            north_queue=self.lanes["north"]["queue"],
            south_queue=self.lanes["south"]["queue"],
            east_queue=self.lanes["east"]["queue"],
            west_queue=self.lanes["west"]["queue"],
            north_wait=sum(self.lanes["north"]["wait_times"]) if self.lanes["north"]["wait_times"] else 0,
            south_wait=sum(self.lanes["south"]["wait_times"]) if self.lanes["south"]["wait_times"] else 0,
            east_wait=sum(self.lanes["east"]["wait_times"]) if self.lanes["east"]["wait_times"] else 0,
            west_wait=sum(self.lanes["west"]["wait_times"]) if self.lanes["west"]["wait_times"] else 0,
            current_phase=self.current_phase.value,
            phase_duration=self.phase_duration,
            emergency_active=(self.emergency_vehicle is not None),
            emergency_lane=self.emergency_vehicle["lane"] if self.emergency_vehicle else None,
            total_wait_time=self.total_wait_time,
            vehicles_served=self.vehicles_served,
            total_congestion_score=float(total_queue)
        )
        return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.steps += 1
        reward_info = self._apply_action(action.action)
        self._generate_traffic()
        self._process_movement()
        self._update_wait_times()
        
        reward = self._calculate_reward(action, reward_info)
        done = self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, reward_info

    def _apply_action(self, action: Action) -> Dict:
        info = {"penalty": 0.0, "success": False}
        
        if self.in_transition:
            self.transition_timer -= 1
            self.phase_duration += 1
            if self.transition_timer <= 0:
                self.current_phase = self.next_phase
                self.in_transition = False
                self.phase_duration = 0
            return info

        if action == ActionType.SWITCH_NS_GREEN:
            if self.current_phase != TrafficPhase.NS_GREEN:
                self.in_transition = True
                self.transition_timer = self.yellow_time
                self.next_phase = TrafficPhase.NS_GREEN
                self.current_phase = TrafficPhase.EW_YELLOW
            else:
                info["penalty"] += 0.1 # Redundant switch
        
        elif action == ActionType.SWITCH_EW_GREEN:
            if self.current_phase != TrafficPhase.EW_GREEN:
                self.in_transition = True
                self.transition_timer = self.yellow_time
                self.next_phase = TrafficPhase.EW_GREEN
                self.current_phase = TrafficPhase.NS_YELLOW
            else:
                info["penalty"] += 0.1
                
        elif action == ActionType.KEEP_PHASE or action == ActionType.EXTEND_GREEN:
            self.phase_duration += 1
            
        elif action == ActionType.EMERGENCY_OVERRIDE:
            if self.emergency_vehicle:
                # Force switch to emergency lane
                target = TrafficPhase.NS_GREEN if self.emergency_vehicle["lane"] in ["north", "south"] else TrafficPhase.EW_GREEN
                if self.current_phase != target:
                    self.current_phase = target
                    self.phase_duration = 0
                    info["success"] = True
                else:
                    info["penalty"] += 0.05
            else:
                info["penalty"] += 0.2 # No emergency to override for
        
        if not self.in_transition:
            self.phase_duration += 1
            
        return info

    def _generate_traffic(self):
        for lane_name, lane in self.lanes.items():
            # Poisson arrival
            if random.random() < self.arrival_rates[lane_name]:
                lane["queue"] += 1
                lane["wait_times"].append(0)
        
        # Emergency vehicle spawn (1% per step)
        if not self.emergency_vehicle and random.random() < 0.02:
            lane_name = random.choice(list(self.lanes.keys()))
            self.emergency_vehicle = {"lane": lane_name, "time_arrived": self.steps}
            # Ensure at least one vehicle is in that lane
            if self.lanes[lane_name]["queue"] == 0:
                self.lanes[lane_name]["queue"] = 1
                self.lanes[lane_name]["wait_times"].append(0)

    def _process_movement(self):
        # Determine which lanes are green
        green_lanes = []
        if self.current_phase == TrafficPhase.NS_GREEN:
            green_lanes = ["north", "south"]
        elif self.current_phase == TrafficPhase.EW_GREEN:
            green_lanes = ["east", "west"]
            
        # Process departures (2 vehicles per step if green)
        for lane_name in green_lanes:
            lane = self.lanes[lane_name]
            departures = min(lane["queue"], 2)
            for _ in range(departures):
                wait = lane["wait_times"].pop(0)
                self.total_wait_time += wait
                self.vehicles_served += 1
                lane["queue"] -= 1
                
                # Check if emergency vehicle cleared
                if self.emergency_vehicle and self.emergency_vehicle["lane"] == lane_name:
                    # In a real sim, we'd track the specific vehicle. 
                    # Here we'll just say if queue is low or random chance.
                    if random.random() < 0.5 or lane["queue"] == 0:
                        self.emergency_vehicle = None

    def _update_wait_times(self):
        for lane in self.lanes.values():
            for i in range(len(lane["wait_times"])):
                lane["wait_times"][i] += 1

    def _calculate_reward(self, action: ActionType, info: Dict) -> float:
        total_queue = sum(l["queue"] for l in self.lanes.values())
        avg_wait = sum(sum(l["wait_times"]) for l in self.lanes.values()) / (total_queue + 1)
        
        reward = 0.0
        reward -= 0.01 * total_queue
        reward -= 0.001 * avg_wait
        
        # Serving vehicles
        reward += 0.05 * self.vehicles_served / (self.steps + 1)
        
        # Emergency
        if self.emergency_vehicle:
            reward -= 0.5
            # If emergency lane is green
            if (self.emergency_vehicle["lane"] in ["north", "south"] and self.current_phase == TrafficPhase.NS_GREEN) or \
               (self.emergency_vehicle["lane"] in ["east", "west"] and self.current_phase == TrafficPhase.EW_GREEN):
                reward += 0.2
        
        reward -= info["penalty"]
        if info.get("success"):
            reward += 1.0
            
        return float(reward)

    def state(self) -> Dict:
        return {
            "lanes": self.lanes,
            "current_phase": self.current_phase,
            "emergency_vehicle": self.emergency_vehicle,
            "steps": self.steps
        }
