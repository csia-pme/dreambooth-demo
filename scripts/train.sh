#!/bin/bash

if [ -z "$SUBJECT_NAME" ]; then
  echo "SUBJECT_NAME is empty"
else

  IMAGE_SIZE=$(yq '.train.size' ../params.yaml)
  TRAIN_STEPS=$(yq '.train.steps' ../params.yaml)
  CHECKPOINT_STEPS=$(yq '.train.checkpoint' ../params.yaml)
  LEARNING_RATE=$(yq '.train.rate' ../params.yaml)

  #export MODEL_NAME="stabilityai/stable-diffusion-2"
  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
  export INSTANCE_DIR="../data/$SUBJECT_NAME"
  export OUTPUT_DIR="../model/$SUBJECT_NAME"
  export CLASS_DIR="../class"

  mkdir -p $INSTANCE_DIR
  mkdir -p $OUTPUT_DIR

  echo "Launching training for $SUBJECT_NAME using $MODEL_NAME"

  accelerate launch --num_processes=1 --gpu_ids=0 ./diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of $SUBJECT_NAME person" \
    --class_prompt="a photo of a person" \
    --resolution=$IMAGE_SIZE \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=$LEARNING_RATE \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=100 \
    --max_train_steps=$TRAIN_STEPS \
    --train_text_encoder \
    --checkpointing_steps=$CHECKPOINT_STEPS \
    --num_train_epochs=1 
fi