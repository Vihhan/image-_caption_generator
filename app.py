import gradio as gr
from model import generate_caption

def predict(image):
    caption = generate_caption(image)
    return caption

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Caption Generator",
    description="Upload an image and get a descriptive AI-generated caption!"
)

iface.launch()
