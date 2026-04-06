import gradio as gr
import numpy as np
import time
from env import TrafficEnvironment
from models import Action, ActionType

def simulate_run(task_id):
    env = TrafficEnvironment(seed=42)
    obs = env.reset()
    
    logs = []
    frames = []
    
    for i in range(20): # Short run for demo
        # Simple heuristic
        if obs.emergency_active:
            action = Action(action=ActionType.EMERGENCY_OVERRIDE)
        else:
            action = Action(action=ActionType.KEEP_PHASE)
            
        next_obs, reward, done, info = env.step(action)
        
        # Create a visual representation
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
