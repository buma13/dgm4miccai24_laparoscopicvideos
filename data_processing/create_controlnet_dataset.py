
import os
from ultralytics import YOLO
import cv2
import torch
from tqdm import tqdm
import csv
import jsonlines
import pandas as pd
import numpy as np
import os
import json
import shutil

CLASS_NAME_MAPPING = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

CLASS_COLOR_MAPPING = {
    0: (127, 127, 127),
    1: (210, 140, 140),
    2: (255, 114, 114),
    3: (231, 70, 156),
    4: (186, 183, 75),
    5: (170, 255, 0),
    6: (255, 85, 0),
    7: (255, 0, 0),
    8: (255, 255, 0),
    9: (169, 255, 184),
    10: (255, 160, 165),
    11: (0, 50, 128),
    12: (111, 74, 0),
}

### Creates ControlNet dataset folder structure by annotating the dataset used in StableDiffusion.

SD_DATASET_PATH = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/custom_cholec_combined/train/" # path to dataset used for StableDiffusion training
DEST_PATH = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/controlnet/" # path to save destination
cholect50_path = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/CholecT50"
cholec80_path = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/cholec80"

model = YOLO('/mnt/projects/mlmi/dmcaf_laparoscopic/cholecseg8k-yolov8/runs/segment/train/weights/best.pt') # insert path of finetuned YOLOv8 model here

vids = os.listdir(SD_DATASET_PATH)
with jsonlines.open(os.path.join(DEST_PATH,'train.jsonl'), 'w') as writer:
    for idx, v in enumerate(vids):
        if "VID" in v:
            os.makedirs(os.path.join(DEST_PATH,"images",v), exist_ok=True)
            results = model(SD_DATASET_PATH+v, save=False,
                            device='cuda', stream=True, show_labels=False, verbose=False)
            masks_path = os.path.join(DEST_PATH,"conditioning_images/{}/".format(v))
            images_path = os.path.join(DEST_PATH,"images/{}/".format(v))
            if not os.path.exists(masks_path):
                os.makedirs(masks_path, exist_ok=True)
                os.makedirs(images_path, exist_ok=True)

            metadata_df = pd.read_csv(os.path.join(
                SD_DATASET_PATH, "metadata.csv"))

            for i,result in tqdm(enumerate(results)):
                img_path = "/".join(result.path.split("/")[-2:])
                text = metadata_df.loc[metadata_df['file_name'] == img_path]["text"].tolist()[0]
                if text == "preparation": continue # Skip preparation frames
                if os.path.exists(os.path.join(cholect50_path, "labels", v+".json")):
                        with open(os.path.join(cholect50_path, "labels", v+".json")) as f:
                            idx = int(img_path.split("/")[1].split(".")[0])
                            label_file = json.load(f)["annotations"]
                            labels = label_file[str(idx)]
                            n_instruments = len([label[1] for label in labels if label[1]!=-1])
                elif os.path.exists(os.path.join(cholec80_path, "tool_annotations", f"video{v.strip('VID')}-tool.txt")):
                        label_path = os.path.join(cholec80_path, "tool_annotations", f"video{v.strip('VID')}-tool.txt")
                        df = pd.read_csv(label_path, delimiter="\t")
                        df['Frame'] //= 25
                        df = df.drop('Frame', axis=1)
                        idx = int(img_path.split("/")[1].split(".")[0])
                        n_instruments = sum(df.iloc[i].tolist())
                else:
                    print(f"{img_path}: no annotation file found! skipping image")
                    continue

                if result.masks:
                    masks = result.masks.data
                    classes = result.boxes.cls.to(torch.int64).cpu().tolist()
                    h, w = masks.shape[-2:]
                    seg_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    seg_mask[:] = CLASS_COLOR_MAPPING[0]
                    for mask, cls in zip(masks, classes):
                        color = CLASS_COLOR_MAPPING.get(cls, CLASS_COLOR_MAPPING[0])
                        seg_mask[mask.bool().cpu().numpy()] = color
                else:
                    continue
                line_dict = {"text": text, "image": 'images/'+img_path,
                                "conditioning_image": 'conditioning_images/'+img_path}
                writer.write(line_dict)
                seg_mask_bgr = cv2.cvtColor(seg_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(masks_path+img_path.split("/")[1], seg_mask_bgr)
                shutil.copyfile(os.path.join(SD_DATASET_PATH, img_path), os.path.join(DEST_PATH, "images", img_path))

