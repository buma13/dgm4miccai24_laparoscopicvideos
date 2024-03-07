
export MODEL_DIR="<StableDiffusion path>"
export OUTPUT_DIR="<path to save resulting ControlNet model>"
export TRAIN_DIR="<training dataset directory>"

accelerate launch models/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$TRAIN_DIR \
 --resolution=128 \
 --learning_rate=1e-5 \
 --num_train_epochs=100 \
 --train_batch_size=256 \
 --dataloader_num_workers=16 \
 --checkpointing_steps=1000 \
 #--resume_from_checkpoint="latest" \
 #--report_to="wandb" 
 #--validation_image 
 #--validation_prompt