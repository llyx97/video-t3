#!/bin/bash
#SBATCH --job-name=mmlm_submit
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100 
#SBATCH --gres=gpu:8
#SBATCH --partition=compute,h100bldg40
#SBATCH --mem=0 

export NNODES=1 
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export BATCH_SIZE=1
# Set global batch size
GLOBAL_BATCH_SIZE=64
# Calculate gradient accumulation steps
export GRADIENT_ACCU_STEPS=$((GLOBAL_BATCH_SIZE / NNODES / GPUS_PER_NODE / BATCH_SIZE))
export MASTER_PORT=32456
export CPUS_PER_TASK=100 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export QUOTA=reserved
export NCCL_NET_PLUGIN=none 
export DATA_ROOT="/home/lilei/Open-LLaVA-NeXT/data/t3"
export IMAGE_ROOT="/home/lilei/Open-LLaVA-NeXT/data"
export DATA_NAME=$1 
echo "data name: $DATA_NAME"

export DATA_PATH=${DATA_NAME}.json
#
# hotpotqa_16k-llava_next_per1000_lRatio25
# keyframe_qa_16k-llava_next_per1000_lRatio25
# tempqa_v2_long_order-tempqa_v2_long_attribute-llava_next_per1000_lRatio25
export BASE_LR=1e-5  
export VIT_LR=2e-6 

export SAVE_PATH=${DATA_NAME}_baseLR${BASE_LR}_vitLR${VIT_LR}_$(date +"%Y%m%d_%H%M%S")
echo "save to $SAVE_PATH"



SRUN_ARGS=${SRUN_ARGS:-""}
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 
# 
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
export S2_CKPT="lmms-lab/LongVA-7B"
export PROMPT_VERSION="qwen_1_5"
export VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"

srun --jobid $SLURM_JOBID  \
    bash -c 'torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    longva/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${S2_CKPT} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_ROOT}/${DATA_PATH} \
    --image_folder ${IMAGE_ROOT} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type unires \
    --bf16 True \
    --run_name $SAVE_PATH \
    --output_dir "./checkpoints/${SAVE_PATH}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa'