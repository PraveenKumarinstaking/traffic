import os
import json
import time
from env import TrafficEnvironment
from models import Action, ActionType, Observation
from grader import run_grading
import openai

# Configuration for OpenAI-compatible client
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = openai.OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

def get_agent_action(obs: Observation, task_description: str) -> Action:
    """
    Calls the LLM to decide the next action.
    In a real competition environment, this would be a prompt.
    For the baseline, we use a heuristic but keep the OpenAI structure.
    """
    # Prompt construction
    prompt = f"Task: {task_description}\nState: {obs.model_dump_json()}\nAction (one of Action enum):"
    
    try:
        # Heuristic baseline to save costs/time, but structure is here
        # In a real run, uncomment the LLM call
        """
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        prediction = response.choices[0].message.content.strip()
        return Action(prediction)
        """
        
        # Simple heuristic policy for baseline:
        if obs.emergency_active:
            return Action(action=ActionType.EMERGENCY_OVERRIDE)
        
        # Switch if queue is large on the other phase
        ns_queue = obs.north_queue + obs.south_queue
        ew_queue = obs.east_queue + obs.west_queue
        
        if obs.current_phase == "NS_GREEN":
            if ew_queue > ns_queue + 5:
                return Action(action=ActionType.SWITCH_EW_GREEN)
            return Action(action=ActionType.KEEP_PHASE)
        else:
            if ns_queue > ew_queue + 5:
                return Action(action=ActionType.SWITCH_NS_GREEN)
            return Action(action=ActionType.KEEP_PHASE)
            
    except Exception as e:
        return Action(action=ActionType.KEEP_PHASE)

def main():
    tasks = [
        ("congestion_relief", "Task 1 — Basic Congestion Relief: Reduce total queue length."),
        ("fair_scheduling", "Task 2 — Fair Phase Scheduling: Balance traffic flow."),
        ("emergency_priority", "Task 3 — Emergency Vehicle Prioritization: Prioritize emergency vehicles.")
    ]

    print("[START]")
    
    all_scores = {}

    for task_id, task_desc in tasks:
        env = TrafficEnvironment(seed=42) # Deterministic for grading
        obs = env.reset()
        history = []
        
        task_start_time = time.time()
        
        for step in range(100):
            action = get_agent_action(obs, task_desc)
            next_obs, reward, done, info = env.step(action)
            
            # [STEP] output for telemetry
            print(f"[STEP] Task: {task_id}, Step: {step}, Action: {action.action.value}, Reward: {reward:.4f}")
            
            history.append((obs, action, reward))
            obs = next_obs
            
            if done:
                break
        
        score = run_grading(history, task_id)
        all_scores[task_id] = score
        print(f"Task {task_id} Score: {score:.4f}")

    print("[END]")
    print(f"Final results: {json.dumps(all_scores)}")

if __name__ == "__main__":
    main()
