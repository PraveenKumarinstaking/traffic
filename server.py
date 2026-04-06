from fastapi import FastAPI, Body
import gradio as gr
import numpy as np
from env import TrafficEnvironment
from models import Action, ActionType, Observation, Reward
from typing import Dict, Any

# Initialize the global environment
global_env = TrafficEnvironment(seed=42)

# Initialize FastAPI
app = FastAPI(title="Emergency-Aware Traffic Signal Control API")

# --- OpenEnv API Endpoints ---

@app.post("/reset", response_model=Observation)
async def reset():
    """Resets the environment and returns the initial observation."""
    obs = global_env.reset()
    return obs

@app.post("/step")
async def step(action: Action):
    """Executes a step in the environment."""
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    """Returns the current state of the environment."""
    return global_env.state()

# --- Gradio UI ---

def simulate_run(task_id):
    # Use a fresh env for local demo to not interfere with global API state
    demo_env = TrafficEnvironment(seed=42)
    obs = demo_env.reset()
    
    logs = []
    frames = []
    
    for i in range(20):
        if obs.emergency_active:
            action = Action(action=ActionType.EMERGENCY_OVERRIDE)
        else:
            action = Action(action=ActionType.KEEP_PHASE)
            
        next_obs, reward, done, info = demo_env.step(action)
        
        frame = f"""
        Intersection Status (Step {i}):
        Phase: {obs.current_phase}
        
              [N: {obs.north_queue}]
                 |
        [W: {obs.west_queue}] --+-- [E: {obs.east_queue}]
                 |
              [S: {obs.south_queue}]
              
        Emergency Active: {'YES' if obs.emergency_active else 'NO'}
        """
        frames.append(frame)
        logs.append(f"Step {i}: Action={action.action.value}, Reward={reward:.2f}")
        obs = next_obs
        if done: break
        
    return "\n".join(frames), "\n".join(logs)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚦 Emergency-Aware Traffic Signal Control")
    gr.Markdown("An OpenEnv compatible simulation of an intelligent 4-way intersection.")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["congestion_relief", "fair_scheduling", "emergency_priority"],
            value="congestion_relief",
            label="Select Task"
        )
        run_btn = gr.Button("▶️ Run Simulation", variant="primary")
        
    with gr.Row():
        viz = gr.Textbox(label="Visual Simulation", lines=12, interactive=False)
        output_logs = gr.Textbox(label="Step Logs", lines=12, interactive=False)
        
    run_btn.click(simulate_run, inputs=[task_dropdown], outputs=[viz, output_logs])

# --- Mounting Gradio ---
# Mount at /dashboard to prevent shadowing the API routes at the root
app = gr.mount_gradio_app(app, demo, path="/dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
