#!bin/bash

echo "ls"
ls
echo "ls ../"
ls ../
echo "pwd"
pwd

if [ -z "$SUBJECT_NAME" ]; then
  echo "SUBJECT_NAME is empty"
else

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/$SUBJECT_NAME"
mkdir -p INSTANCE_DIR
export OUTPUT_DIR="../model/$SUBJECT_NAME"
mkdir -p OUTPUT_DIR
#export CLASS_DIR="../class"

  accelerate launch ./diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of $SUBJECT_NAME the person" \
  --class_prompt="a photo of a person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10 
# --with_prior_preservation --prior_loss_weight=1.0 \
# --class_data_dir=$CLASS_DIR \
# --num_class_images=100 \
# --checkpointing_steps=100
fi