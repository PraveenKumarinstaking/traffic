import os
import json
import time
import sys
from env import TrafficEnvironment
from models import Action, ActionType, Observation
from grader import run_grading
import openai

# --- Configuration ---
# Variables will be initialized in main() to ensure [START] is printed first
API_BASE_URL = None
API_KEY = None
MODEL_NAME = None
client = None

def get_agent_action(obs: Observation, task_description: str) -> Action:
    """
    Calls the LLM to decide the next action based on the state.
    """
    prompt = f"""
    Task: {task_description}
    Current Observation:
    - Queues (N/S/E/W): {obs.north_queue}/{obs.south_queue}/{obs.east_queue}/{obs.west_queue}
    - Current Phase: {obs.current_phase}
    - Phase Duration: {obs.phase_duration}
    - Emergency Active: {obs.emergency_active}
    - Emergency Lane: {obs.emergency_lane}
    
    Available Actions: {', '.join([a.value for a in ActionType])}
    
    Respond ONLY with the name of the ActionType that should be taken.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a traffic controller LLM. Output only the ActionType."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0
        )
        action_text = response.choices[0].message.content.strip()
        
        # Validating output against ActionType enum
        for action_type in ActionType:
            if action_type.value in action_text:
                return Action(action=action_type)
        
        print(f"[WARNING] Invalid LLM response: '{action_text}'. Falling back to heuristic.")
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {e}. Falling back to heuristic.")

    # --- Heuristic Fallback (Original Logic) ---
    if obs.emergency_active:
        return Action(action=ActionType.EMERGENCY_OVERRIDE)
    
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

def main():
    global API_BASE_URL, API_KEY, MODEL_NAME, client
    
    tasks = [
        ("congestion_relief", "Task 1 - Basic Congestion Relief: Reduce total queue length."),
        ("fair_scheduling", "Task 2 - Fair Phase Scheduling: Balance traffic flow."),
        ("emergency_priority", "Task 3 - Emergency Vehicle Prioritization: Prioritize emergency vehicles."),
        ("throughput_maximization", "Task 4 - Throughput Maximization: Maximize total vehicles served per step.")
    ]

    # Mandatory [START] tag - MUST be first
    print("[START]")
    
    try:
        # Initialize configuration after [START]
        # WARNING: Hardcoding API keys is not recommended for production.
        # Use environment variables or secrets management instead.
        API_BASE_URL = os.environ.get("API_BASE_URL")
        API_KEY = os.environ.get("API_KEY")
        MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

        client = openai.OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        all_scores = {}

        for task_id, task_desc in tasks:
            env = TrafficEnvironment(seed=42) # Deterministic for grading
            obs = env.reset()
            history = []
            
            for step in range(100):
                action = get_agent_action(obs, task_desc)
                next_obs, reward, done, info = env.step(action)
                
                # [STEP] output for telemetry
                # Format reward to 4 decimal places
                # json.dumps handles lowercase booleans automatically
                print(f"[STEP] Task: {task_id}, Step: {step}, Action: {action.action.value}, Reward: {reward:.4f}, Done: {json.dumps(done)}")
                
                history.append((obs, action, reward))
                obs = next_obs
                
                if done:
                    break
            
            score = run_grading(history, task_id)
            all_scores[task_id] = score
            print(f"Task {task_id} Score: {score:.4f}")

        print(f"Final results: {json.dumps(all_scores)}")

    except Exception as e:
        print(f"[ERROR] Runtime exception during inference: {str(e)}")
        # Optionally re-raise or handle
    finally:
        # Mandatory [END] tag - MUST always be printed even on failure
        print("[END]")

if __name__ == "__main__":
    main()
