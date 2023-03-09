#!bin/bash

if [ -z "$SUBJECT_NAME" ]; then
  echo "SUBJECT_NAME is empty"
else

  MODEL_NAME=$(yq '.train.model_name' ../params.yaml)

  INSTANCE_DIR="../data/prepared"
  OUTPUT_DIR="../models"
  mkdir -p $INSTANCE_DIR
  mkdir -p $OUTPUT_DIR

  INSTANCE_PROMPT=$(yq '.train.instance_prompt' ../params.yaml)
  CLASS_PROMPT=$(yq '.train.class_prompt' ../params.yaml)
  IMAGE_SIZE=$(yq '.train.image_size' ../params.yaml)
  LEARNING_RATE=$(yq '.train.learning_rate' ../params.yaml)
  TRAIN_STEPS=$(yq '.train.steps' ../params.yaml)


  echo "Launching training using $MODEL_NAME"
  echo "Image size $IMAGE_SIZE"
  echo "Learning rate $LEARNING_RATE"
  echo "Train steps $TRAIN_STEPS"

  accelerate launch --num_processes=1 --gpu_ids=0 ./diffusers/examples/dreambooth/train_dreambooth.py \
    --pretrained_model_name_or_path="$MODEL_NAME"  \
    --instance_data_dir="$INSTANCE_DIR" \
    --class_data_dir="../class" \
    --output_dir="$OUTPUT_DIR" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks person" \
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
    --num_train_epochs=1 
fi