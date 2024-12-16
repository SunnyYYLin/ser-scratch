import gradio as gr
from .events import load_model, predict, MODELS

def load_demo():
    with gr.Blocks() as demo:
        gr.Markdown('# Speech Emotion Recognition')
        gr.Markdown('---\n## Load model')
        
        model = gr.State(value=None)
        
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label='model',
                    choices=MODELS,
                    value='CNN',
                    interactive=True
                )
            with gr.Column():
                load_btn = gr.Button(
                    value='Load model',
                    interactive=True,
                )
        gr.Markdown('---\n## Predict')
        with gr.Row():
            audio_input = gr.Audio(
                label='audio',
                type='numpy',
                sources=['microphone', 'upload'],
                interactive=True
            )
                    
            with gr.Column():
                emo_output = gr.Textbox(
                    label='emotion',
                    type='text',
                    interactive=True
                )
        with gr.Row():
            pred_btn = gr.Button(
                value='Predict',
                interactive=True
            )
            
        load_btn.click(
            fn=load_model,
            inputs=[model_name],
            outputs=[model]
        )
        
        pred_btn.click(
            fn=predict,
            inputs=[model, audio_input],
            outputs=[emo_output]
        )
    
    return demo