import os
import ultralytics
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import PIL
import os
import numpy as np
import pandas as pd
import tqdm
import cv2
from torchvision import transforms


### creates ControlNet dataset for evaluation and saves original conditioning in yolov8 format
### same dataset is also used for fidelity eval and factual correctness eval

def mask2yolo(mask_path, dest_path):

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_LINEAR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygon = []
    for c in contours:
        if len(c) > len(polygon):
            polygon = c.squeeze().astype(float)
    if len(polygon) > 0:
        polygon = ultralytics.utils.ops.scale_coords((640, 640), polygon, (640, 640), normalize=True)
        polygon = polygon.flatten().tolist()
        polygon = "0 " + ' '.join(map(str, polygon))

        with open(dest_path, 'w') as file:
            file.write(polygon)

def build_controlnet_datasets(sd_path, controlnet_path, data_path, dest_path, model_name, num_inference_steps=100, guidance_scale=3, device="cuda:2", chunk_size=500):
    
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_path,
                                                                controlnet=controlnet,
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True, safety_checker=None)
    pipe.to(device)

    
    df = pd.read_json(path_or_buf=os.path.join(data_path, "test.jsonl"), lines=True)

    ctrl_images = []
    prompts = []
    for i in range(len(df)):
        row = df.iloc[i] 
        prompts.append(row['text'])
        ctrl_images.append(row['conditioning_image'])
    
    prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
    ctrl_chunks = [ctrl_images[i:i + chunk_size] for i in range(0, len(ctrl_images), chunk_size)]

    
    idx = 0
    fake_img_list = []
    ctrl_img_list = []
    for prompts, ctrls in zip(prompt_chunks, ctrl_chunks):
        seg_imgs_pil = [PIL.Image.open(os.path.join(data_path,ctrl)).convert('RGB') for ctrl in ctrls]
        images = pipe(prompts,
                    image=seg_imgs_pil,
                    height=128, width=128,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    ).images
        for idx, image in enumerate(images):
            transform = transforms.ToTensor()
            tensor_image = transform(image) * 255
            fake_img_list.append(tensor_image)
        for x in seg_imgs_pil:
            ctrl_img_list.append(x)
    
    
    for idx, (img, ctrl, ctrl_path) in enumerate(zip(fake_img_list, ctrl_img_list, ctrl_images)):
        img = PIL.Image.fromarray((img.permute(
            1, 2, 0).cpu().detach().numpy()).astype(np.uint8), mode="RGB")
        os.makedirs(os.path.join(dest_path, "images", model_name), exist_ok=True)
        os.makedirs(os.path.join(dest_path, "labels", model_name), exist_ok=True)
        img.save(os.path.join(dest_path, "images", model_name,"{}.png".format(str(idx).zfill(6))))
        mask2yolo(os.path.join(data_path, ctrl_path), os.path.join(dest_path, "labels", model_name,"{}.txt".format(str(idx).zfill(6))))



if __name__ == "__main__":
    CONTROLNET_PATH = "" # path to ControlNet model
    SD_PATH = "" # path to StableDiffusion model
    testset_path = "" # path to testset containing metadata.csv and conditioning images
    dest_path = "" # path to save destination
    model_name = "controlnet"

    build_controlnet_datasets(SD_PATH, CONTROLNET_PATH, testset_path, dest_path, model_name)