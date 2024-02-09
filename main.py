import mediapy as media
import random
import sys
import torch

import gradio as gr

import webbrowser
from threading import Thread

from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo',
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
)

pipe = pipe.to('cuda')


def greet(positive_prompt, negative_prompt):
    seed = random.randint(0, sys.maxsize)

    num_inference_steps = 100

    images = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    ).images

    return positive_prompt, seed, images[0]


app = gr.Interface(fn=greet, inputs=['text', 'text'], outputs=[gr.Textbox(label='Prompt'), gr.Number(label='Seed'), gr.Image(label='GAN Image')])
Thread(target = app.launch).start()
Thread(target = lambda: webbrowser.open_new('http://127.0.0.1:7860')).start()
