import pandas as pd
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms


### Create dataset for StableDiffusion evaluation from a metadata.csv file.


def build_fid_datasets(model_paths, data_path, dest_path, num_inference_steps, guidance_scale, checkpoint_steps=None, epoch_steps=None):
    # sample 10.000 images and their prompts from the real dataset and save them
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    prompts = metadata["text"].tolist()
    prompt_chunks = [prompts[i:i + 500] for i in range(0, len(prompts), 500)]

    for model in model_paths:
        if checkpoint_steps:
            unet = UNet2DConditionModel.from_pretrained(model+f"checkpoint-{checkpoint_steps}/unet")
        else:
            unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")

        text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder")

        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")

        fake_img_list = []
        for chunk in prompt_chunks:
            pipe = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, safety_checker=None, requires_safety_checker=False)
            pipe.to("2")
            images = pipe(prompt=chunk, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                          height=128, width=128).images
            for idx, image in enumerate(images):
                transform = transforms.ToTensor()
                tensor_image = transform(image) * 255
                fake_img_list.append(tensor_image)

        fake_images = torch.stack(fake_img_list)


        if checkpoint_steps:
            model_name = str(checkpoint_steps//epoch_steps)+model.split("/")[-2]
        else:
            model_name = model.split("/")[-1]

        save_path = os.path.join(
            dest_path, model_name, f"{num_inference_steps}_{guidance_scale}")

        if not (os.path.isdir(save_path)):
            os.makedirs(save_path)

        for i in range(fake_images.size(0)):
            img = Image.fromarray((fake_images[i, :, :, :].permute(
                1, 2, 0).cpu().detach().numpy()).astype(np.uint8), mode="RGB")
            img.save(os.path.join(save_path, '{}.png'.format(str(i).zfill(6))))


if __name__ == "__main__":
    model_paths = [] # list of paths of StableDiffusion models to evaluate
    testset_path = "" # path to testset with metadata.csv
    dest_path = "" # path to save destination

    guidance_scale = 3
    num_inference_steps = 100

    build_fid_datasets(model_paths, testset_path, dest_path,num_inference_steps, guidance_scale)
