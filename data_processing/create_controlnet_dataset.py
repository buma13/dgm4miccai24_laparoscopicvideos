
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

### Creates ControlNet dataset folder structure by annotating the dataset used in StableDiffusion.


SD_DATASET_PATH = "" # path to dataset used for StableDiffusion training
DEST_PATH = "" # path to save destination
cholect50_path = ""
cholec80_path = ""

model = YOLO('') # insert path of finetuned YOLOv8 model here

vids = os.listdir(SD_DATASET_PATH)
with jsonlines.open(os.path.join(DEST_PATH,'train.jsonl'), 'w') as writer:
    for idx, v in enumerate(vids):
        if "VID" in v:
            os.makedirs(os.path.join(DEST_PATH,"images",v), exist_ok=True)
            results = model(SD_DATASET_PATH+v, save=False,
                            device='cuda:1', stream=True, show_labels=False, verbose=False)
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
                    #option to crossval number of predicted masks with cholect50 tool presence labels, if they're not equal skip the image       
                    # if n_instruments != len(result.masks):
                    #     continue

                    masks = result.masks.data
                    tool_mask = (torch.any(masks, dim=0).int()
                                 * 255).cpu().numpy()
                    
                # option to include negative examples containing no tools
                # elif n_instruments==0:
                #     tool_mask = np.zeros((128, 128))    
                else:
                    continue   
                line_dict = {"text": text, "image": 'images/'+img_path,
                                "conditioning_image": 'conditioning_images/'+img_path}
                writer.write(line_dict)
                cv2.imwrite(masks_path+img_path.split("/")[1], tool_mask)
                shutil.copyfile(os.path.join(SD_DATASET_PATH, img_path), os.path.join(DEST_PATH, "images", img_path))
                
