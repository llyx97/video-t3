export SIZE_FACTOR=8 
export MAX_PIXELS=401408  #602112  = 768 * 28 * 28 
export NPROC_PER_NODE=8 
RATIO=${1:-"VideoOnly"}

export FILE_NAME="T366K+VRatio2.0.noimage.valid.ms.videoOnly.jsonl"

export DATA_DIR="/home/lilei/LLaMA-Factory/converted_data/"
export NFRAMES=8    
# --resume_from_checkpoint "save/72b_lora_lRatio2_video100k_1000step_from450/qwen2-vl-72b-instruct/v8-20241122-135916/checkpoint-550" 
echo $FILE_NAME 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
  --model_type qwen2-vl-72b-instruct \
  --model_id_or_path /home/lilei/Qwen2-VL-72B-Instruct  \
  --sft_type lora \
  --freeze_vit true \
  --learning_rate 1e-5 --weight_decay 0.03  \
  --dataset $DATA_DIR/$FILE_NAME \
    --val_dataset $DATA_DIR/ms.val.jsonl --save_steps 100 --max_steps 1000 --gradient_accumulation_steps 8 --batch_size 1 --gradient_checkpointing true   \
    --output_dir save/72b_lora_T3-66K_video${RATIO}  --max_length 5120 --dataloader_num_workers 2  --warmup_ratio 0.03 \
    --deepspeed default-zero3  --dtype bf16 
