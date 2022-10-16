import gradio as gr
from argparse import ArgumentParser
import torch
import logging
from diffusers import StableDiffusionPipeline
from multiprocessing import Lock
from datetime import datetime
from PIL import Image
import os
import json
from static import *

logging.basicConfig(level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S",
                    format='%(asctime)s - %(levelname)s - %(name)s\n%(message)s')

def parse_args():
    parser = ArgumentParser()
    bool_fn = lambda x: 'y' in x.lower()
    parser.add_argument('--test', type=bool_fn, default='yes')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default='7890')
    parser.add_argument('--device', type=str, default=None, help='if you have multiple devices, specify it as 0, 1, etc.')
    parser.add_argument('--auth', type=str, default=None, help='fill it with yours in huggingface account to download diffusion model weights')

    args = parser.parse_args()
    return args

args = parse_args()
lock = Lock()
pipe = None
logger = logging.getLogger('app')
if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
args.device = 'cuda'
if not args.test:
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=args.auth,
                                                   revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    torch.backends.cudnn.benchmark = True

def save_args_to_json(path, path_to_img, timestamp, height, width, samples, steps, scale, seed):
    with open(path, 'w') as file:
        json.dump(
            {
                'image_path': path_to_img,
                'time': timestamp,
                'height': height,
                'width': width,
                'num': samples,
                'steps': steps,
                'scale': scale,
                'seed': seed
            }, file, indent=4
        )
    logger.info(f'dump args to {path}')

def infer(prompt, height=512, width=512, samples=4, steps=42, scale=7.5, seed=2147483647):
    global args
    global lock
    global pipe
    logger.info(f"input args:\nprompt={prompt}\tsize={height}x{width}\tnum={samples}\tstep={steps}\tscale={scale}\tseed={seed}")
    images = []
    if args.test:
        logger.info("you are in testing mode")
        for i in range(samples):
            images.append(Image.open('images/unsafe.png'))
    else:
        lock.acquire()
        images_list = pipe(
            [prompt] * samples,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=torch.Generator(device=args.device).manual_seed(seed),
        )
        now = datetime.now().strftime("%y-%m-%d-%H_%M_%S")
        for i, image in enumerate(images_list["sample"]):
            image.save(f'images/{now}_{i}.png')
            save_args_to_json(f'images/{now}_{i}.json', f'images/{now}_{i}.png', now, height, width, samples, steps, scale, seed)
            images.append(image)
        lock.release()

    return images



if __name__ == '__main__':

    block = gr.Blocks(css=CSS)
    with block:
        gr.HTML(HTML)
        with gr.Group():
            with gr.Box():
                with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    btn = gr.Button("Generate image").style(
                        margin=False,
                        rounded=(False, True, True, False),
                        full_width=False,
                    )

            gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
            ).style(grid=[3], height="auto")

            with gr.Row(elem_id="advanced-options"):
                samples = gr.Slider(label="Images", minimum=1, maximum=6, value=4, step=1)
                height = gr.Slider(label="height", minimum=16, maximum=1024, value=512, step=16)
                width = gr.Slider(label="width", minimum=16, maximum=1024, value=512, step=16)
                steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )

            standard_input = lambda : [text, height, width, samples, steps, scale, seed]

            ex = gr.Examples(examples=EXAMPLES, fn=infer, inputs=standard_input(), outputs=gallery, cache_examples=False)
            ex.dataset.headers = [""]
            text.submit(infer, inputs=standard_input(), outputs=gallery)
            btn.click(infer, inputs=standard_input(), outputs=gallery)

            gr.HTML(WARNING_HTML)


    block.queue(concurrency_count=10, max_size=10).launch(max_threads=64, server_name=args.host, server_port=args.port)