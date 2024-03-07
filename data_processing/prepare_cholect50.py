from dataloader import CholecT50
import csv
from tqdm import tqdm
import cv2
import os
import pandas as pd
import shutil


### Extracts prompts from CholecT50 and creates training dataset folder structure for StableDiffusion.

def prepare_cholect50(cholect50_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    cholect50 = CholecT50(dataset_dir=cholect50_path,
                          dataset_variant="cholect50_full")

    header = ['file_name', 'text']
    if os.path.isfile(os.path.join(save_path, 'metadata.csv')):
        print("metadata.csv already exists!")
        return
    with open(os.path.join(save_path, 'metadata.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for video in cholect50.train_set_list:
            for idx, sample in tqdm(enumerate(video)):
                file_name = sample[2]

                text = sample[3]
                text = text.replace(",", " ")
                text = text.replace("null_verb", "")
                text = text.replace("null_target", "")
                # for tool in tools:
                #     if tool not in text:
                #         text = "{} and {}".format(tool, text)

                writer.writerow([file_name, " ".join(text.split())])
                img = cv2.imread(os.path.join(cholect50_path,  "videos/", file_name))
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
                os.makedirs(os.path.join(save_path, file_name.split("/")[0]), exist_ok=True)
                cv2.imwrite(os.path.join(save_path, file_name), img)
                
                
def create_testset(data_path, dest_path):
    
    if os.path.isfile(os.path.join(dest_path, 'metadata.csv')):
        print("metadata.csv already exists!")
        return
    
    os.makedirs(dest_path, exist_ok=True)
    train_metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    test_metadata = train_metadata.sample(10000, random_state=42)

    for row in test_metadata.iterrows():
        os.makedirs(os.path.join(dest_path, row[1]['file_name'].split("/")[0]), exist_ok=True)
        shutil.move(os.path.join(data_path,row[1]['file_name']), os.path.join(dest_path,row[1]['file_name'].split("/")[1]))


    test_metadata["file_name"] = test_metadata["file_name"].apply(lambda x: x.replace("/", ""))
    test_metadata.to_csv(os.path.join(dest_path, "metadata.csv"), index=False)

    train_metadata = train_metadata.drop(test_metadata.index)
    train_metadata.to_csv(os.path.join(data_path, "metadata.csv"), index=False)




if __name__ == "__main__":
    cholect50_path = "data/CholecT50"
    save_path = ""
    prepare_cholect50(cholect50_path, save_path)
    
    test_path = ""
    create_testset(save_path, test_path)

