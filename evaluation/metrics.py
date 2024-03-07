import torch
from pytorch_msssim import ssim
from cleanfid import fid
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import itertools
from transformers import CLIPImageProcessor, CLIPModel
from cmmd_pytorch.main import compute_cmmd


def calc_pairwise_ssim(images_path):
    scores = []
    for prompt_folder in os.listdir(images_path):
        image_paths = [os.path.join(images_path, prompt_folder, x) for x in os.listdir(
            os.path.join(images_path, prompt_folder))]
        x = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        x = torch.from_numpy(np.array(x)).permute(0, 3, 1, 2).float()
        for subset in itertools.combinations(x, 2):
            ssim_score = ssim(subset[0].unsqueeze(
                dim=0), subset[1].unsqueeze(dim=0), data_range=255)
            scores.append(ssim_score)
    scores_np = np.array(scores)
    return np.mean(scores_np), np.std(scores_np)


def calc_fid(image_folder1, image_folder2, devices="0", clip=False):
    if clip:
        fid_score = fid.compute_fid(image_folder1, image_folder2, device="cuda",
                                    num_workers=1, dataset_res=128, model_name="clip_vit_b_32")
    else:
        fid_score = fid.compute_fid(image_folder1, image_folder2, device="cuda",
                                    num_workers=1, dataset_res=128)
    return fid_score


def calc_kid(image_folder1, image_folder2, devices="0", clip=False):
    kid_score = fid.compute_kid(image_folder1, image_folder2, device="cuda",
                                num_workers=1, dataset_res=128)
    return kid_score


def calc_cmmd(image_folder1, image_folder2, devices="0", custom_clip=None):
    return compute_cmmd(image_folder1, image_folder2)
