import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown('# Speech Emotion Recognition')
    gr.Markdown('---\n## Load model')
    
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(
                label='model',
                choices=['CNN', 'BiLSTM'],
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
            sources=['microphone'],
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
                
demo.launch()