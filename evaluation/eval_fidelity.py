from metrics import *
import pandas as pd
import torch
import json
from build_fid_datasets import build_fid_datasets

### script for running fidelity evaluation, saves a json of the results at the end


TESET_PATH = ""
SYNTHSET_PATHS = []

results = []
for s in SYNTHSET_PATHS:
    fid_folder = ""

    cmmd_ = calc_cmmd(TESET_PATH, fid_folder).item()
    v3_fid = calc_fid(TESET_PATH, fid_folder)
    clip_fid = calc_fid(TESET_PATH, fid_folder, clip=True)
    kid = calc_kid(TESET_PATH, fid_folder)
    
    r = {'Experiment': s.split("/")[-2:] , 'CMMD': round(cmmd_, 3), 'V3_FID': round(v3_fid,3), 'CLIP_FID': round(clip_fid,3), 'KID': round(kid,5)}
    print(r)
    results.append(r)
    torch.cuda.empty_cache()

with open("evaluation/metric_results.json", "w") as f:
    json.dump(results, f)



