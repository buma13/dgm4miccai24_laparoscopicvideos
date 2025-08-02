import cv2
import os
import pandas as pd
import csv
import string
from tqdm import tqdm
import sys
import ast


### Extracts prompts from Cholec80 and creates training dataset folder structure for StableDiffusion

def prepare_cholec80(cholec80_path, save_path, video_ids):
    with open(os.path.join(save_path, "metadata.csv"), 'a', encoding='UTF8') as f:
        # header = ['file_name', 'text']
        writer = csv.writer(f)
        # writer.writerow(header)
        for vid_id in video_ids:
            phase_annotations = pd.read_table(os.path.join(
                cholec80_path, "phase_annotations/", "video{}-phase.txt".format(vid_id)))
            tool_annotations = pd.read_table(os.path.join(
                cholec80_path, "tool_annotations/", "video{}-tool.txt".format(vid_id)))

            for frame_number in tqdm(tool_annotations["Frame"]):
                cap = cv2.VideoCapture(os.path.join(
                    cholec80_path, "videos/", "video{}.mp4".format(vid_id)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
                res, frame = cap.read()
                if res:
                    # make phase names consistent with CholecT50
                    phase = phase_annotations["Phase"][frame_number].lower()
                    if phase == "calottriangledissection":
                        phase = "carlot-triangle-dissection"
                    elif phase == "clippingcutting":
                        phase = "clipping-and-cutting"
                    elif phase == "gallbladderdissection":
                        phase = "gallbladder-dissection"
                    elif phase == "gallbladderpackaging":
                        phase = "gallbladder-packaging"
                    elif phase == "cleaningcoagulation":
                        phase = "cleaning-and-coagulation"
                    elif phase == "gallbladderretraction":
                        phase = "gallbladder-extraction"

                    tools = [list(tool_annotations)[idx] if x ==
                             1 else None for idx, x in enumerate(tool_annotations.loc[tool_annotations["Frame"] == frame_number].values.flatten().tolist())]
                    tools = [x.lower() for x in tools if x is not None]
                    text = " and ".join(tools).strip() + " in " + \
                        phase if len(tools) > 0 else phase

                    vid_path = os.path.join(save_path, "VID"+vid_id)
                    os.makedirs(vid_path, exist_ok=True)
                    frame_path = os.path.join(vid_path, str(
                        frame_number).rjust(6, "0") + ".png")
                    frame = cv2.resize(frame, (128, 128),
                                       interpolation=cv2.INTER_LINEAR)

                    cv2.imwrite(frame_path, frame)
                    writer.writerow([frame_path.strip(save_path), text])


if __name__ == "__main__":
    cholec80_path = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/cholec80"
    save_path = "/mnt/projects/mlmi/dmcaf_laparoscopic/dataset/custom_cholec_combined/train"

    # videos that are not included in CholecT50
    video_ids = ["03", "07", "09", "11", "16",
                "17", "19", "20", "21", "24",
                "28", "30", "33", "34", "37",
                "38", "39", "41", "44", "45",
                "46", "53", "54", "55", "58",
                "59", "61", "63", "64", "67",
                "69", "71", "72", "76", "77"]
    print(video_ids)
    prepare_cholec80(cholec80_path, save_path, video_ids)



