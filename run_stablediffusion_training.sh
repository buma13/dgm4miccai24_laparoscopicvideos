export MODEL_NAME="runwayml/stable-diffusion-v1-5"

### CholecT50 runs
export TRAIN_DIR="/home/bwlki/ivan/laparodiffusion/data/sd_datasets/train_cholect50+80"
export OUTPUT_DIR="/home/bwlki/ivan/laparodiffusion/models/sd_checkpoints/11LR1e-5-cholect50+80-txt"
export RUN_NAME="LR1e-5-cholect50+80-txt"
export EPOCHS=10
export LR=1e-5

accelerate launch models/train_text_to_image.py \
  --run_name=$RUN_NAME \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=128 \
  --train_batch_size=128 \
  --dataloader_num_workers=16 \
  --use_ema \
  --gradient_checkpointing \
  --unfreeze_text_encoder \
  --mixed_precision="bf16" \
  --checkpointing_steps=4850 \
  --num_train_epochs=$EPOCHS \
  --resume_from_checkpoint="latest" \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_epochs=1 \
  --report_to="wandb" \
  --validation_prompts "carlot-triangle-dissection" "grasper grasp gallbladder in carlot-triangle-dissection" \
                        "grasper grasp gallbladder and hook in carlot-triangle-dissection" \
                        "grasper and hook in carlot-triangle-dissection" \
                        "grasper retract gallbladder and hook dissect gallbladder in gallbladder-dissection"