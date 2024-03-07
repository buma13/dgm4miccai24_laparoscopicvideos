# Interactive Generation of Laparoscopic Videos with Diffusion Models
---
## Prerequisites
### Python packages
`pip install -r requirements.txt`

### Github Repos
- ControlVideo https://github.com/YBYBZhang/ControlVideo
- CMMD-pytorch https://github.com/sayakpaul/cmmd-pytorch (for fidelity evaluation)
- Rendezvouz (RDV) https://github.com/CAMMA-public/rendezvous (for factual correctness evaluation)

### Datasets
- CholecT45, CholecT50 and Cholec80: available at http://camma.u-strasbg.fr/datasets
### Model weigths
All our resulting model weights will be released after the anonymous review period of MICCAI 2024.
Those include weights for:
- StableDiffusion
- ControlNet
- RDV for 128x128 images (for evaluation)
- YOLOv8 model (for surgical tool segmentation)

## Training Instructions
There are some placeholders in all scripts, replace them with your own values (such as paths, prompts, parameters etc.) beforehand.
### Stable Diffusion Finetuning
1. Download CholecT50 and Cholec80.
2. Run `data_processing/prepare_cholect50.py` and `data_processing/prepare_cholec80.py` to create the training dataset.
3. Run training with `run_stablediffusion_training.sh`.



### ControlNet Training
1. Run `evaluation/create_controlnet_datasets.py` (once for train and once for test).
2. Run training with `run_controlnet_training.sh`.

## Inference Instructions
- StableDiffusion inference script: `model_scripts/sd_inference.py`.
- ControlNet inference script: `model_scripts/sd_inference.py`.
- For ControlVideo, clone the repository and replace models in inference scripts with our custom models.

## Evaluation Instructions
### Fidelity
1. Run `evaluation/fidelity_eval_datasets.py` to generate synthetic dataset from the `metadata.csv` file created during `data_processing/prepare_cholect50.py`.</br>
For ControlNet run `evaluation/controlnet_eval_datasets.py`, generates the data from a `test.jsonl` file that can be created by running `data_processing/create_controlnet_dataset.py`.
2. Run `evaluation/eval_fidelity.py`, providing the path to the synthetic data to be evaluted.
3. Results are saved to a `.json` file.

### Factual Corectness using RDV
1. Run `evaluation/create_rdv_testset.ipynb` notebook.
2. Clone RDV and modify dataloader to include the created testset.
3. Run RDV test script.

### ControlNet F1 Score
1. `evaluation/controlnet_eval_datasets.py` generates segmentation labels compatible with YOLOv8
2. Do an eval run according to YOLOV8 documentation (using our finetuned YOLOv8 model): https://docs.ultralytics.com/tasks/segment/#models