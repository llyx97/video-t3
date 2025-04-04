CONFIG=${1:-"examples/train_full/qwen2vl_full_t3_7b.yaml"}
echo "Running with config: $CONFIG" 
FORCE_TORCHRUN=1 llamafactory-cli train $CONFIG 
