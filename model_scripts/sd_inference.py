from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


FINETUNED_SD_PATH = "/mnt/projects/mlmi/dmcaf_laparoscopic/models/StableDiffusion"

unet = UNet2DConditionModel.from_pretrained(FINETUNED_SD_PATH, subfolder="unet")

text_encoder = CLIPTextModel.from_pretrained(
    FINETUNED_SD_PATH, subfolder="text_encoder")

tokenizer = CLIPTokenizer.from_pretrained(FINETUNED_SD_PATH, subfolder="tokenizer")

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, safety_checker=None, requires_safety_checker=False)
pipe.to("cuda")

prompts = ["grasper and specimenbag in gallbladder-packaging with bleeding"] # List of Promts

save_path = "generated_images"
for prompt in prompts:
    images = pipe(prompt=prompt, num_inference_steps=100, guidance_scale=3,
                height=128, width=128, num_images_per_prompt=6).images
    images_array = []
    for idx, image in enumerate(images):
        image = ImageOps.expand(image, border=3, fill='white')
        images_array.append(image)

    image_grid(images_array, rows=2, cols=3).save("{}/{}.png".format(save_path, prompt))
