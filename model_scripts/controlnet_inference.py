from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image, ImageOps
import os
import numpy as np

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


CONTROLNET_PATH = ""
STABLEDIFFUSION_PATH = ""
controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    STABLEDIFFUSION_PATH, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None)
pipe.to("cuda")

prompts = [] # list of text prompts
ctrl_images = [] # paths to conditioning images



idx = 0
images_array = []
for prompt, ctrl in zip(prompts, ctrl_images):

    seg_img_pil = Image.open(os.path.join(ctrl)).convert('RGB').resize((128,128))
    image = pipe(prompt,
                    image=seg_img_pil,
                    height=128, width=128, num_inference_steps=100, guidance_scale=3, num_images_per_prompt=1,
                    ).images[0]

    image = np.concatenate((seg_img_pil, image), axis=0)
    image = Image.fromarray(image)
    image = ImageOps.expand(image, border=3, fill='white')
    images_array.append(image)
    

save_path = ""
image_grid(images_array, rows=1, cols=3).save(
"/{}.png".format(save_path,prompt))

