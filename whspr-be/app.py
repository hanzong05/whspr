"""
HuggingFace Spaces entry point.
Mounts the FastAPI app onto a minimal Gradio interface.
HF Spaces (Gradio SDK) serves on port 7860.
"""
import gradio as gr
from main import app  # FastAPI app

with gr.Blocks() as demo:
    gr.Markdown("# WHSPR API")
    gr.Markdown("Backend is running. Use the API endpoints directly.")

# Mount Gradio UI at /ui, FastAPI routes stay at their original paths
app = gr.mount_gradio_app(app, demo, path="/ui")
