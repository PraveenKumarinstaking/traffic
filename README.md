---
title: Emergency-Aware Traffic Signal Control
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Emergency-Aware Traffic Signal Control Environment

## Overview
This repository contains a real-world, intelligent traffic signal control environment built following the **OpenEnv** specification. The environment simulates a single 4-way intersection where an agent must optimize traffic flow, maintain fairness across lanes, and prioritize emergency vehicles (Ambulances, Fire Trucks, etc.).

Traffic congestion costs cities billions of dollars annually and delays emergency services. This environment provides a platform to train and evaluate agents on managing these critical urban systems.

## Real-World Relevance
- **Congestion Mitigation**: Reducing average wait time and queue lengths.
- **Emergency Priority**: Ensuring life-saving vehicles reach their destinations with zero avoidable delay.
- **Carbon Footprint**: Less idling time translates to lower CO2 emissions.

## Action Space
The agent can perform the following actions at each step:
- `KEEP_PHASE`: Maintain the current signal state.
- `SWITCH_NS_GREEN`: Transitions to North-South green (via yellow).
- `SWITCH_EW_GREEN`: Transitions to East-West green (via yellow).
- `EXTEND_GREEN`: Extends the current green phase duration.
- `EMERGENCY_OVERRIDE`: Immediately switches signals to accommodate an active emergency vehicle.

## Observation Space
Typed `Observation` model including:
- `north_queue`, `south_queue`, `east_queue`, `west_queue`: Number of vehicles waiting.
- `north_wait`, `south_wait`, `east_wait`, `west_wait`: Cumulative wait time per lane.
- `current_phase`: Current signal phase (`NS_GREEN`, `EW_GREEN`, etc.).
- `emergency_active`: Boolean flag for emergency presence.
- `total_congestion_score`: Total vehicles currently in queues.

## Task Descriptions

### Task 1 — Basic Congestion Relief (Easy)
**Objective**: Reduce total queue length over a short horizon.
**Success Criteria**: Lower queue buildup and average wait time.

### Task 2 — Fair Phase Scheduling (Medium)
**Objective**: Balance traffic flow across all four directions.
**Success Criteria**: Avoid starving any lane; maintain low variance in wait times.

### Task 3 — Emergency Vehicle Prioritization (Hard)
**Objective**: Detect and prioritize emergency vehicles with minimal delay.
**Success Criteria**: Zero delay for emergency lanes without causing total gridlock.

## Grading Methodology
Grading is programmatically calculated on a scale of [0.0, 1.0]:
- **Congestion Score**: Based on average queue length vs. baseline.
- **Fairness Score**: Based on the standard deviation of wait times across lanes.
- **Emergency Score**: Based on the time-to-clear for emergency vehicles.

## Setup & Usage

### Local Development
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pydantic numpy openai gradio
   ```
3. Run the environment baseline:
   ```bash
   python inference.py
   ```

### Docker
Build and run the container:
```bash
docker build -t traffic-env .
docker run -p 7860:7860 traffic-env
```

## Baseline Scores
The baseline heuristic agent achieves the following approximate scores:
- **Task 1**: 0.85
- **Task 2**: 0.78
- **Task 3**: 0.92

## Deployment
This environment is optimized for deployment on **Hugging Face Spaces** using the provided `Dockerfile` and `app.py` for visualization.
